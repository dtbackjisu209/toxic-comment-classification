"""Evaluation helpers for toxic comment classification."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device) -> Dict[str, float]:
	model.eval()
	all_predictions: List[int] = []
	all_labels: List[int] = []
	total_loss = 0.0
	total_batches = 0

	for batch in dataloader:
		batch = {key: value.to(device) for key, value in batch.items()}
		labels = batch.pop("labels")
		outputs = model(**batch, labels=labels)
		loss = outputs["loss"]
		logits = outputs["logits"]

		total_loss += float(loss.item()) if loss is not None else 0.0
		total_batches += 1
		predictions = torch.argmax(logits, dim=1)

		all_predictions.extend(predictions.detach().cpu().tolist())
		all_labels.extend(labels.detach().cpu().tolist())

	average_loss = total_loss / max(total_batches, 1)
	return {
		"loss": average_loss,
		"accuracy": accuracy_score(all_labels, all_predictions) if all_labels else 0.0,
		"precision": precision_score(all_labels, all_predictions, zero_division=0) if all_labels else 0.0,
		"recall": recall_score(all_labels, all_predictions, zero_division=0) if all_labels else 0.0,
		"f1": f1_score(all_labels, all_predictions, zero_division=0) if all_labels else 0.0,
	}

