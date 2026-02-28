# Amazon Videogame Search System with RLHF and NER

A hybrid intelligent search system for videogame product discovery that combines semantic retrieval (FAISS), Named Entity Recognition (NER), and Reinforcement Learning from Human Feedback (RLHF) with a pointwise reward model trained from pairwise human preferences and a PPO policy. The system operates entirely locally without cloud dependencies on a corpus of 9,999 Amazon videogame products.

---

## Key Features

- **Semantic Search**: FAISS-based vector similarity search with `all-MiniLM-L6-v2` (384-dim embeddings)
- **NER Enhancement**: Zero-shot product attribute extraction for query intent detection (genre, platform, franchise, category)
- **Pointwise Reward Model**: 435K-parameter MLP trained from human A/B pairwise preferences (val_acc ≈ 0.91–0.96)
- **PPO Policy**: Learned reranking policy trained via Proximal Policy Optimization with KL-adaptive beta
- **Interactive A/B Collection**: CLI tool to collect human preferences between policy and baseline rankings
- **Human Ground Truth**: Interactive ground truth builder with manual relevance annotation (no FAISS leakage)
- **Reproducible Evaluation**: nDCG@10, Recall@10, MRR, MAP@10 with paired t-test and Bonferroni correction
- **Local Processing**: Full pipeline runs on GPU (CUDA) or CPU, no cloud dependencies

---

## Evaluated Methods

The system benchmarks 5 ranking configurations on the same test query set:

| Method | Description |
|---|---|
| **Baseline (FAISS)** | Pure cosine similarity semantic search |
| **NER-Enhanced** | Baseline + attribute bonus for genre/platform/franchise matches |
| **Reward-Only** | Baseline candidates reranked by pointwise reward model score |
| **RLHF (PPO)** | Baseline candidates reranked by PPO-trained policy |
| **Full Hybrid** | NER reranking + PPO policy combined |

### Latest Results (135 A/B preferences, 15 test queries, interactive ground truth)

| Method | nDCG@10 | Recall@10 | MRR | MAP@10 | Δ% vs Baseline | p-val |
|---|---|---|---|---|---|---|
| Baseline (FAISS) | 0.8497 | 1.0000 | 0.8000 | 0.7437 | — | — |
| NER-Enhanced | 0.7788 | 0.8321 | **0.8667** | 0.6719 | -8.3% | 0.045* |
| **Reward-Only** | **0.8817** | **1.0000** | **0.9167** | **0.7762** | **+3.8%** | 0.533 |
| RLHF (PPO) | 0.7616 | 1.0000 | 0.6833 | 0.6250 | -10.4% | 0.152 |
| Full Hybrid | 0.7424 | 0.8321 | 0.8778 | 0.5981 | -12.6% | 0.019* |

> **Key findings**: Reward-Only achieves the best nDCG@10 (+3.8%). NER improves MRR (+8.3%) but degrades nDCG. PPO requires more preference data to converge reliably. Significance at p<0.05 requires larger test set or higher Δ.

---

## Project Structure

```
proyecto-ia-rag-rl/
├── src/
│   ├── unified_system_v2.py           # Core system: 9,999 products, FAISS + NER + RLHF
│   ├── data/
│   │   ├── loader.py                  # Amazon JSONL dataset loader
│   │   ├── canonicalizer.py           # Title + description fusion
│   │   ├── cache_manager.py           # Embedding + system cache
│   │   └── vector_store.py            # FAISS index manager
│   ├── ranking/
│   │   ├── baseline_ranker.py         # FAISS cosine similarity
│   │   ├── ner_enhanced_ranker.py     # FAISS + attribute bonus scoring
│   │   └── rl_ranker_fixed.py         # Legacy RLHF ranker
│   ├── enrichment/
│   │   └── ner_zero_shot_optimized.py # Zero-shot NER attribute extraction
│   ├── query/
│   │   └── understanding.py           # Query intent detection
│   └── rlhf/
│       ├── __init__.py
│       ├── pointwise_reward_model.py  # 435K-param MLP reward model
│       ├── policy_model.py            # Attention-based policy (heads=4, layers=2)
│       ├── ppo_trainer.py             # PPO with KL-adaptive beta
│       ├── rlhf_pipeline.py           # End-to-end RLHF pipeline
│       ├── preference_collector.py    # A/B preference data loader
│       └── tensor_utils.py            # Embedding utilities
├── data/
│   ├── raw/                           # Raw Amazon JSONL product metadata
│   ├── cache/                         # Embedding cache (9,999 products × 384 dims)
│   ├── preferences/
│   │   └── preferences.jsonl          # Human A/B pairwise preferences
│   ├── rlhf_checkpoints/
│   │   ├── reward_model.pt            # Trained reward model checkpoint
│   │   └── policy_model.pt            # Trained PPO policy checkpoint
│   └── interactions/
│       ├── queries.txt                # 51 unique search queries
│       ├── ground_truth_REAL.json     # Human-annotated relevance judgments
│       ├── train_queries.json         # Train split (~55%)
│       ├── test_queries.json          # Test split (~45%, no leakage)
│       └── split_info.json            # Split metadata (seed=42)
├── results/                           # Evaluation results (JSON + TXT per run)
├── main.py                            # Entry point: init / interactivo / rlhf / experimento
├── train_pointwise_reward.py          # Train reward model from A/B preferences
├── evaluate_methods.py                # 5-method evaluation with statistical tests
├── ground_truth_builder.py            # Build ground truth (--mode auto/interactive)
├── split_queries.py                   # Train/test split with leakage check
├── extract_queries_from_interactions.py  # Extract queries from interaction logs
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA GPU recommended (CPU supported but slower)
- 8GB+ RAM
- ~5GB disk for model cache

### 2. Installation

```bash
git clone <repo>
cd proyecto-ia-rag-rl

python -m venv ambiente
ambiente\Scripts\activate      # Windows
# source ambiente/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Data Preparation

Download `meta_Video_Games.jsonl` from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) and place it in `data/raw/`.

Expected product format:
```json
{
  "parent_asin": "B001234567",
  "title": "Product name",
  "description": ["Detailed description"],
  "features": ["Feature 1", "Feature 2"],
  "price": 29.99,
  "average_rating": 4.5,
  "rating_number": 1234,
  "main_category": "Video Games"
}
```

### 4. System Initialization

```bash
# First run: builds FAISS index and embedding cache (~2-5 min for 9,999 products)
python main.py init

# Verify system
python main.py stats
```

---

## Main Commands

### Build Ground Truth

```bash
# Automatic mode (FAISS-based, threshold 0.55) — fast but has circular evaluation risk
python ground_truth_builder.py --mode auto --threshold 0.55

# Interactive mode (human-annotated) — recommended for valid evaluation
python ground_truth_builder.py --mode interactive
```

### Split Queries (Train / Test)

```bash
# Creates train/test split with seed=42, checks for leakage
python split_queries.py

# Force overwrite existing split
python split_queries.py --force
```

### Collect Human Preferences (A/B)

```bash
# Interactive session: compare policy vs baseline rankings
python main.py interactivo
```

The system shows two ranked lists (A and B) for each query. You select the preferred ranking. Each session generates ~10-20 preferences stored in `data/preferences/preferences.jsonl`.

Format:
```json
{
  "query": "survival games",
  "ranking_a_ids": ["B00...", ...],
  "ranking_b_ids": ["B00...", ...],
  "ranking_a_type": "policy",
  "ranking_b_type": "baseline",
  "preference": "B",
  "preferred_type": "baseline"
}
```

### Train Reward Model

```bash
python train_pointwise_reward.py
```

Reads all A/B preferences from `data/preferences/preferences.jsonl`, extracts pairwise comparisons using rank-swap logic, and trains the pointwise reward model. Reports val_accuracy and chosen-rejected gap.

Current best: **1,472 pairs from 135 preferences → val_acc=0.909, chosen-rejected=+0.960**

### Train PPO Policy

```bash
# Default: 50 queries, 5 epochs
python main.py rlhf --ppo

# Custom: 50 queries, 10 epochs
python main.py rlhf --ppo 50 10
```

Loads the trained reward model and runs PPO with KL-adaptive beta to learn a reranking policy. Saves versioned checkpoints: `policy_v0_before_cycleN.pt`, `policy_v1_after_cycleN.pt`.

### Run Evaluation

```bash
python evaluate_methods.py
```

Evaluates all 5 methods on the test query set. Reports nDCG@10, Recall@10, MRR, MAP@10 with paired t-test and Bonferroni correction. Results saved to `results/evaluation_YYYYMMDD_HHMMSS.json`.

### Full Recommended Workflow

```bash
# 1. Build ground truth interactively
python ground_truth_builder.py --mode interactive

# 2. Split queries
python split_queries.py

# 3. Collect preferences (repeat for more data)
python main.py interactivo

# 4. Train reward model
python train_pointwise_reward.py

# 5. Train PPO policy
python main.py rlhf --ppo 50 10

# 6. Evaluate
python evaluate_methods.py
```

---

## System Architecture

### RLHF Pipeline

```
Human A/B Preferences (preferences.jsonl)
        │
        ▼
Pair Extraction (rank-swap logic)
  chosen: ranks higher in winner AND lower in loser
        │
        ▼
PointwiseRewardModel (MLP: 384→256→1, input=1536)
  Input: [q_emb; prod_emb; q_emb*prod_emb; |q_emb-prod_emb|]
  Loss: margin ranking loss (margin=0.5)
        │
        ▼
PPO Training
  Policy: Transformer (emb=384, hidden=128, heads=4, layers=2)
  Reward signal: reward_model(query, product) per candidate
  KL constraint: adaptive beta (target_kl=0.02)
        │
        ▼
Evaluation: nDCG@10, Recall@10, MRR, MAP@10
```

### NER Intent Detection

The NER module detects intent types from queries and applies bonus scoring:

- **franchise**: "minecraft", "zelda", "pokemon", "mario" → boosts exact franchise matches
- **platform**: "ps5", "xbox", "nintendo switch" → boosts platform-specific products
- **genre**: "mmorpg", "fps", "rpg", "survival" → boosts genre matches
- **category**: "videogames", "gaming" → general category boost

Bonus weight: 15% over base FAISS score. NER improves MRR (+8.3% observed) but can reduce nDCG by boosting irrelevant products when intent is ambiguous.

### Reward Model Architecture

```
Input: concat([q_emb, p_emb, q_emb * p_emb, |q_emb - p_emb|])  # 384*4 = 1,536 dims
       → Linear(1536, 256) → ReLU → Dropout(0.1)
       → Linear(256, 128) → ReLU → Dropout(0.1)
       → Linear(128, 1) → scalar reward score

Parameters: 435,457
Training: margin ranking loss, lr=2e-4, batch=32, early stopping (patience=7)
```

---

## Evaluation Design

### Ground Truth Construction

Two modes are supported:

- **Auto mode** (`--threshold 0.55`): Uses FAISS cosine similarity to auto-label relevant products. Fast but creates circular evaluation (baseline nDCG→1.0). Use only for debugging.
- **Interactive mode** (recommended): Human annotator reviews top-30 FAISS candidates per query and manually selects relevant products. Breaks circularity and produces valid evaluation data.

Current ground truth: **45 queries, 219 relevants, avg 4.9/query, min 1 / max 10**

### Split Strategy

- Seed: 42 (fixed, never change)
- Test fraction: 45% (~20 queries)
- Leakage check enforced: train and test queries must not overlap
- Train queries are used for PPO training; test queries are held out for evaluation

### Statistical Testing

- Paired t-test (per-query nDCG@10 differences)
- Bonferroni correction for multiple comparisons
- Significance threshold: p < 0.05

> **Note on sample size**: With n=15 test queries, a Δ=+3.8% nDCG requires ~n=30 for p<0.05. The Reward-Only result is directionally consistent but not yet statistically significant.

---

## Troubleshooting

**Cache corruption**
```bash
del data\cache\unified_system_v2.pkl
python main.py init
```

**PPO reward negative** — reward model and policy are misaligned. Retrain PPO from scratch after collecting more preferences:
```bash
del data\rlhf_checkpoints\policy_model.pt
python main.py rlhf --ppo 50 10
```

**Baseline nDCG = 1.0** — ground truth was built with auto mode (FAISS leakage). Rebuild with interactive mode:
```bash
python ground_truth_builder.py --mode interactive
python split_queries.py --force
```

**Low reward model accuracy (<0.85)** — insufficient or noisy preference data. Collect more A/B sessions:
```bash
python main.py interactivo   # 20-30 more comparisons
python train_pointwise_reward.py
```

---

## Performance

| Stage | Time |
|---|---|
| System initialization (first run) | ~2-5 min (9,999 products) |
| Embedding cache load (subsequent) | ~2 sec |
| Reward model training (1,472 pairs) | ~9 sec (GPU) |
| PPO training (50 queries × 10 epochs) | ~15 sec (GPU) |
| Evaluation (15 queries × 5 methods) | ~3 sec (GPU) |

---

## Research Context

This system implements a full RLHF loop for information retrieval:

1. **Retrieval**: Dense retrieval with FAISS + sentence transformers
2. **Preference Learning**: Pointwise reward model from pairwise human judgments
3. **Policy Optimization**: PPO over the reward signal with KL regularization
4. **Evaluation**: IR metrics on human-annotated ground truth

Key findings from experiments:
- Reward-only reranking consistently outperforms the baseline when trained on 100+ preferences
- NER improves precision-at-1 (MRR) but can hurt full-ranking quality (nDCG) in ambiguous domains
- PPO requires aligned preference and ground truth signals to converge — preference pairs collected during early (weak) policy phases bias learning toward baseline behavior
- Statistical significance with IR metrics requires n≥30 test queries for small effect sizes (Δ<5%)
