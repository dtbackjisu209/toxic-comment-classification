"""PhoBERT sequence classifier backed by Hugging Face pretrained weights."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class PhoBertClassifier(nn.Module):
	def __init__(self, model_name: str = "vinai/phobert-base", num_labels: int = 2) -> None:
		super().__init__()
		self.model_name = model_name
		self.num_labels = num_labels
		self.model = AutoModelForSequenceClassification.from_pretrained(
			model_name,
			num_labels=num_labels,
			ignore_mismatched_sizes=True,
		)
		self.model.config.id2label = {0: "non_toxic", 1: "toxic"}
		self.model.config.label2id = {"non_toxic": 0, "toxic": 1}
		self.model.config.problem_type = "single_label_classification"

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		labels: Optional[torch.Tensor] = None,
	):
		return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

	def save_pretrained(self, save_directory: str | Path) -> None:
		self.model.save_pretrained(save_directory)

	@classmethod
	def from_pretrained(cls, save_directory: str | Path, map_location: Optional[str | torch.device] = None):
		model = cls.__new__(cls)
		nn.Module.__init__(model)
		model.model = AutoModelForSequenceClassification.from_pretrained(save_directory)
		if map_location is not None:
			model.model.to(map_location)
		model.model_name = getattr(model.model.config, "name_or_path", str(save_directory))
		model.num_labels = int(getattr(model.model.config, "num_labels", 2))
		return model

