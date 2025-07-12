# src/core/utils/parsers.py
"""
Reusable text-parsing helpers used across the stack
(RAG evaluator, RLHF reward models, feedback scoring, etc.).

Supports:
• Binary scores (yes/no/unknown)  
• Numeric ranges with safe bounds  
• JSON, list and boolean extraction  
• All functions return Pydantic models for type safety
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ValidationError, Field

from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Domain types
# ------------------------------------------------------------------
class BinaryScore(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class ParseResult(BaseModel):
    """Typed result container for every parser."""
    success: bool = True
    value: Optional[Any] = None
    error: Optional[str] = None
    original: Optional[str] = Field(None, description="Original input text")

    class Config:
        extra = "forbid"


# ------------------------------------------------------------------
# Low-level parsers
# ------------------------------------------------------------------
def parse_binary_score(text: Union[str, Any], *, strict: bool = False) -> BinaryScore:
    """
    Convert free-form text to a canonical BinaryScore.

    Parameters
    ----------
    text   : raw LLM output
    strict : if True, only the exact words 'yes'/'no' are accepted
    """
    if not isinstance(text, str):
        logger.warning("Binary-score input is not str: %s", type(text))
        return BinaryScore.UNKNOWN

    cleaned = text.strip().lower()

    if strict:
        if cleaned == "yes":
            return BinaryScore.YES
        if cleaned == "no":
            return BinaryScore.NO
        return BinaryScore.UNKNOWN

    # Relaxed mode: substring + negation heuristics
    if "yes" in cleaned:
        return BinaryScore.YES
    if "no" in cleaned:
        return BinaryScore.NO

    # Spanish & English negation cues
    negations = {"no", "not", "nunca", "ningún", "ninguna", "incorrect"}
    if any(neg in cleaned for neg in negations):
        return BinaryScore.NO

    return BinaryScore.UNKNOWN


def parse_numeric_range(
    text: str,
    *,
    min_val: Union[int, float],
    max_val: Union[int, float],
    default: Optional[Union[int, float]] = None,
) -> ParseResult:
    """
    Extract the first valid number in a range.

    Returns ParseResult(success=True, value=<float>) or an explanatory error.
    """
    try:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text))
        for n_str in numbers:
            val = float(n_str)
            if min_val <= val <= max_val:
                return ParseResult(value=val, original=text)
        return ParseResult(
            success=False,
            error=f"No value in range {min_val}-{max_val} found",
            original=text,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected numeric parsing error")
        return ParseResult(
            success=False,
            error=str(exc),
            original=text,
        )


def parse_list(text: str, *, delimiter: str = ",") -> ParseResult:
    """Split text by delimiter and trim whitespace."""
    try:
        items = [item.strip() for item in str(text).split(delimiter) if item.strip()]
        return ParseResult(value=items, original=text)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(success=False, error=str(exc), original=text)


def parse_json(text: str) -> ParseResult:
    """Safely parse JSON text into a dict."""
    try:
        data = json.loads(text.strip())
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON object expected")
        return ParseResult(value=data, original=text)
    except (json.JSONDecodeError, ValueError) as exc:
        return ParseResult(success=False, error=str(exc), original=text)


def parse_boolean(text: str) -> ParseResult:
    """Flexible boolean conversion."""
    true_vals = {"true", "yes", "si", "sí", "1", "verdadero", "on"}
    false_vals = {"false", "no", "0", "falso", "off"}
    cleaned = str(text).strip().lower()
    if cleaned in true_vals:
        return ParseResult(value=True, original=text)
    if cleaned in false_vals:
        return ParseResult(value=False, original=text)
    return ParseResult(
        success=False,
        error="Unrecognised boolean string",
        original=text,
    )


# ------------------------------------------------------------------
# Safe wrapper
# ------------------------------------------------------------------
def safe_parse(
    parser: Any,
    text: str,
    *args: Any,
    **kwargs: Any,
) -> ParseResult:
    """
    Wrap any parser with uniform error handling.

    Example
    -------
    result = safe_parse(parse_numeric_range, "score is 4.5", min_val=0, max_val=5)
    """
    try:
        return parser(text, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Parser crashed")
        return ParseResult(success=False, error=str(exc), original=text)