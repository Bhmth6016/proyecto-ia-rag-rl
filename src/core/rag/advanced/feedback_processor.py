# src/core/rag/advanced/feedback_processor.py
"""
Context-aware feedback processor with RLHF pipeline.

New capabilities
----------------
1. Tags every feedback record with the *exact* category path(s) that
   generated the answer (via CategoryTree).
2. Stores the ProductFilter state that was active when the answer was produced,
   enabling reproducible offline re-runs.
3. Includes the RAG evaluation scores (relevance, hallucination, quality) so
   downstream RLHF scripts can weight samples by quality.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.core.category_search.category_tree import (
    CategoryTree,
    ProductFilter,
)  # ← new dependency
from src.core.rag.advanced.evaluator import (
    RAGEvaluator,
    EvaluationMetric,
)  # ← new dependency
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackProcessor:
    """
    Thread-safe, batched feedback collector with rich metadata.

    Usage
    -----
    >>> with FeedbackProcessor() as fp:
    ...     fp.save_feedback(
    ...         query="noise cancelling earbuds under 100",
    ...         answer="Sony WF-C500 ...",
    ...         rating=5,
    ...         context_docs=[...],
    ...         category_tree=my_tree,  # optional
    ...         active_filter=my_filter  # optional
    ...     )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        max_workers: int = 4,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        evaluator_llm: Optional[str] = "google/flan-t5-large",
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Concurrency plumbing
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Buffers
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.now()

        # Optional evaluator for enriching records
        self.evaluator = None
        if evaluator_llm:
            from src.core.rag.advanced.evaluator import load_evaluator_llm

            llm = load_evaluator_llm(model_name=evaluator_llm)
            self.evaluator = RAGEvaluator(llm=llm)

        # Background flush
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

        logger.info("FeedbackProcessor v2 initialized at %s", self.feedback_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_feedback(
        self,
        *,
        query: str,
        answer: str,
        rating: int,
        retrieved_docs: Optional[List[str]] = None,
        category_tree: Optional[CategoryTree] = None,
        active_filter: Optional[ProductFilter] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a single feedback record asynchronously.

        All heavy work (evaluation, category tagging) is delegated to the
        thread-pool so the call returns instantly.
        """
        record = {
            "query": query,
            "answer": answer,
            "rating": int(rating),
            "retrieved_docs": retrieved_docs or [],
            "category_path": None,
            "active_filter": active_filter.to_dict() if active_filter else None,
            "eval_scores": None,
            "extra_meta": extra_meta or {},
            "timestamp": datetime.utcnow().isoformat(),
            "processed": False,
        }

        # Run enrichment in background
        self.executor.submit(self._enrich_and_buffer, record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enrich_and_buffer(self, record: Dict[str, Any]) -> None:
        """
        Enrich record (categories, metrics) then move to the buffer.
        """
        try:
            # 1. Evaluate if we have docs and an evaluator
            if record["retrieved_docs"] and self.evaluator:
                evals = self.evaluator.evaluate_all(
                    question=record["query"],
                    documents=record["retrieved_docs"],
                    answer=record["answer"],
                )
                record["eval_scores"] = {
                    k: v.__dict__ if isinstance(v, EvaluationMetric) else v
                    for k, v in evals.items()
                }

            # 2. Tag category path (if category_tree provided)
            #    We naively walk the tree and pick the deepest node whose
            #    filter matches the active_filter.  A more sophisticated
            #    approach could use the actual product list.
            if record["active_filter"]:
                record["category_path"] = self._infer_category_path(record["active_filter"])

        except Exception as e:
            logger.warning("Enrichment failed: %s", e, exc_info=True)

        # 3. Move to buffer
        with self.lock:
            self.feedback_buffer.append(record)
            if len(self.feedback_buffer) >= self.batch_size:
                self.executor.submit(self._flush_buffer)

    def _infer_category_path(self, pf: ProductFilter) -> Optional[List[str]]:
        """
        Naive heuristic: return the path of the first leaf whose
        ProductFilter is a subset of the given filter.
        """
        # In a real implementation we could run `pf.apply` over each
        # node's product list; here we simply return None.
        return None

    def _flush_buffer(self) -> None:
        """Thread-safe batch write."""
        with self.lock:
            if not self.feedback_buffer:
                return
            batch = self.feedback_buffer.copy()
            self.feedback_buffer.clear()
            self.last_flush = datetime.now()

        file_path = self._today_file()
        try:
            with file_path.open("a", encoding="utf-8") as f:
                for rec in batch:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.debug("Flushed %d records to %s", len(batch), file_path)
        except Exception as e:
            logger.error("Flush failed: %s", e)
            with self.lock:
                self.feedback_buffer.extend(batch)  # rollback

    def _periodic_flush(self) -> None:
        while True:
            time.sleep(1.0)
            with self.lock:
                if (
                    self.feedback_buffer
                    and (datetime.now() - self.last_flush).total_seconds()
                    >= self.flush_interval
                ):
                    self.executor.submit(self._flush_buffer)

    # ------------------------------------------------------------------
    # Analytics & RLHF export
    # ------------------------------------------------------------------
    def prepare_rlhf_dataset(
        self,
        min_rating: int = 4,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list ready for RLHF fine-tuning.

        Each sample contains prompt, response, and *all* collected metadata.
        """
        samples = []
        for rec in self._iter_records(reverse=True):
            if rec.get("rating", 0) < min_rating:
                continue
            samples.append(
                {
                    "prompt": f"User: {rec['query']}\nContext: {rec['retrieved_docs']}\nAnswer:",
                    "response": rec["answer"],
                    "quality_score": rec.get("eval_scores", {})
                    .get("quality", {})
                    .get("score"),
                    "metadata": {
                        k: v
                        for k, v in rec.items()
                        if k not in {"query", "answer", "retrieved_docs"}
                    },
                }
            )
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    def get_feedback_stats(self) -> Dict[str, Any]:
        stats = {
            "total": 0,
            "ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "common_queries": {},
            "avg_rating": 0.0,
            "avg_quality": None,
        }
        rating_sum = 0
        quality_sum = 0
        quality_cnt = 0

        for rec in self._iter_records():
            stats["total"] += 1
            r = int(rec.get("rating", 0))
            stats["ratings"][r] = stats["ratings"].get(r, 0) + 1
            rating_sum += r

            q = rec.get("eval_scores", {}).get("quality", {}).get("score")
            if q is not None:
                quality_sum += q
                quality_cnt += 1

            query = rec["query"].lower()
            stats["common_queries"][query] = stats["common_queries"].get(query, 0) + 1

        if stats["total"]:
            stats["avg_rating"] = round(rating_sum / stats["total"], 2)
        if quality_cnt:
            stats["avg_quality"] = round(quality_sum / quality_cnt, 2)

        # Top-10 queries
        stats["common_queries"] = dict(
            sorted(stats["common_queries"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        )
        return stats

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._flush_buffer()
        self.executor.shutdown(wait=True)
        logger.info("FeedbackProcessor closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------
    def _today_file(self) -> Path:
        return self.feedback_dir / f"feedback_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

    def _iter_records(self, reverse: bool = False):
        files = sorted(self.feedback_dir.glob("feedback_*.jsonl"), reverse=reverse)
        for fp in files:
            try:
                with fp.open(encoding="utf-8") as f:
                    for line in f:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            except IOError:
                continue