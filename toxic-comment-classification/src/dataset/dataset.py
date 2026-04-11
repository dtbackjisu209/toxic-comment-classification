"""PyTorch dataset utilities for toxic comment classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.clean_text import clean_text


def infer_label(value: object) -> int:
	if pd.isna(value):
		raise ValueError("Label value is missing")

	if isinstance(value, bool):
		return int(value)

	if isinstance(value, (int, float)):
		numeric_value = int(value)
		if numeric_value == 0:
			return 0
		if numeric_value in {1, 2}:
			return 1
		raise ValueError(f"Unsupported numeric label: {value!r}. Expected one of 0, 1, 2.")

	normalized = str(value).strip().lower()
	positive_values = {"1", "true", "toxic", "tox", "yes", "y", "abusive", "offensive", "hate"}
	negative_values = {"0", "false", "non-toxic", "non toxic", "clean", "neutral", "no", "n"}

	if normalized in positive_values:
		return 1
	if normalized in negative_values:
		return 0

	try:
		numeric_value = int(float(normalized))
		if numeric_value == 0:
			return 0
		if numeric_value in {1, 2}:
			return 1
		raise ValueError(f"Unsupported numeric label: {value!r}. Expected one of 0, 1, 2.")
	except ValueError as exc:
		raise ValueError(f"Cannot infer label from value: {value!r}") from exc


class ToxicCommentDataset(Dataset):
	def __init__(
		self,
		texts: Sequence[str],
		labels: Sequence[Union[int, str, float]],
		tokenizer,
		max_length: int = 256,
		clean: bool = True,
	) -> None:
		if len(texts) != len(labels):
			raise ValueError("texts and labels must have the same length")

		self.texts = [clean_text(text) if clean else str(text) for text in texts]
		self.labels = [infer_label(label) for label in labels]
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.texts)

	def __getitem__(self, index: int):
		text = self.texts[index]
		label = self.labels[index]
		encoding = self.tokenizer(
			text,
			truncation=True,
			padding="max_length",
			max_length=self.max_length,
			return_tensors="pt",
		)

		item = {key: value.squeeze(0) for key, value in encoding.items()}
		item["labels"] = torch.tensor(label, dtype=torch.long)
		return item


def create_dataloader(
	dataset: Dataset,
	batch_size: int = 16,
	shuffle: bool = False,
	num_workers: int = 0,
) -> DataLoader:
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)

