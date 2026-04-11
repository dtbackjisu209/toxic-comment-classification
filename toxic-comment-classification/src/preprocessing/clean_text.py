"""Text normalization utilities for Vietnamese comments."""

from __future__ import annotations

import re
from typing import Any


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[\w_]+", flags=re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")


def normalize_whitespace(text: Any) -> str:
	if text is None:
		return ""
	return _MULTISPACE_RE.sub(" ", str(text)).strip()


def clean_text(text: Any) -> str:
	text = normalize_whitespace(text).lower()
	if not text:
		return ""

	text = _URL_RE.sub(" ", text)
	text = _MENTION_RE.sub(" ", text)
	text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
	text = re.sub(r"[^\w\sÀ-ỹđĐ_.,!?%:/+-]", " ", text)
	text = _REPEATED_CHAR_RE.sub(r"\1\1", text)
	text = _MULTISPACE_RE.sub(" ", text)
	return text.strip()

