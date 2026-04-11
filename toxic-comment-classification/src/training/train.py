"""Train PhoBERT for toxic vs non-toxic comment classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset.dataset import ToxicCommentDataset, create_dataloader
from src.models.phobert_model import PhoBertClassifier
from src.training.evaluate import evaluate_model
from src.utils.config import TrainingConfig, build_output_paths, default_device, set_seed
from src.utils.data_utils import read_dataset


def split_frame(frame: pd.DataFrame, config: TrainingConfig):
	stratify = frame[config.label_col] if frame[config.label_col].nunique() > 1 else None
	train_frame, val_frame = train_test_split(
		frame,
		test_size=config.train_val_split,
		random_state=config.seed,
		stratify=stratify,
	)
	return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True)


def build_dataset_from_frame(frame: pd.DataFrame, tokenizer, config: TrainingConfig) -> ToxicCommentDataset:
	return ToxicCommentDataset(
		frame[config.text_col].tolist(),
		frame[config.label_col].tolist(),
		tokenizer,
		max_length=config.max_length,
	)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device) -> float:
	model.train()
	total_loss = 0.0
	total_batches = 0

	progress = tqdm(dataloader, desc="Training", leave=False)
	for batch in progress:
		batch = {key: value.to(device) for key, value in batch.items()}
		labels = batch.pop("labels")

		optimizer.zero_grad(set_to_none=True)
		outputs = model(**batch)
		logits = outputs["logits"]
		loss = criterion(logits, labels)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		scheduler.step()

		total_loss += float(loss.item())
		total_batches += 1
		progress.set_postfix(loss=total_loss / max(total_batches, 1))

	return total_loss / max(total_batches, 1)


def save_artifacts(model, tokenizer, output_dir: Path, metrics: dict) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
	with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
		json.dump(metrics, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train PhoBERT for toxic comment classification")
	parser.add_argument("--data-file", type=Path, default=None, help="Single CSV/XLS/XLSX dataset file")
	parser.add_argument("--train-file", type=Path, default=None, help="Train dataset file (CSV/XLS/XLSX)")
	parser.add_argument("--valid-file", type=Path, default=None, help="Validation dataset file (CSV/XLS/XLSX)")
	parser.add_argument("--test-file", type=Path, default=None, help="Test dataset file (CSV/XLS/XLSX)")
	parser.add_argument("--text-col", type=str, default="free_text", help="Text column name")
	parser.add_argument("--label-col", type=str, default="label_id", help="Label column name")
	parser.add_argument("--model-name", type=str, default="vinai/phobert-base", help="HF model name")
	parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--max-length", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--learning-rate", type=float, default=2e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--train-val-split", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	if args.train_file is None and args.data_file is None:
		parser.error("Provide either --train-file (recommended) or --data-file.")

	if args.train_file is not None and args.data_file is not None:
		parser.error("Use only one mode: --train-file/--valid-file/--test-file OR --data-file.")

	if args.valid_file is not None and args.train_file is None:
		parser.error("--valid-file requires --train-file.")

	if args.test_file is not None and args.train_file is None:
		parser.error("--test-file requires --train-file.")

	return args


def main() -> None:
	args = parse_args()
	config = TrainingConfig(
		model_name=args.model_name,
		max_length=args.max_length,
		batch_size=args.batch_size,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		train_val_split=args.train_val_split,
		seed=args.seed,
		text_col=args.text_col,
		label_col=args.label_col,
		output_dir=args.output_dir,
		checkpoint_dir=args.output_dir / "checkpoints",
		log_dir=args.output_dir / "logs",
	)

	build_output_paths(config)
	set_seed(config.seed)
	device = default_device()

	tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
	test_dataset = None

	if args.train_file is not None:
		train_frame = read_dataset(args.train_file, config.text_col, config.label_col)
		if args.valid_file is not None:
			val_frame = read_dataset(args.valid_file, config.text_col, config.label_col)
		else:
			train_frame, val_frame = split_frame(train_frame, config)

		if args.test_file is not None:
			test_frame = read_dataset(args.test_file, config.text_col, config.label_col)
			test_dataset = build_dataset_from_frame(test_frame, tokenizer, config)
	else:
		frame = read_dataset(args.data_file, config.text_col, config.label_col)
		train_frame, val_frame = split_frame(frame, config)

	train_dataset = build_dataset_from_frame(train_frame, tokenizer, config)
	val_dataset = build_dataset_from_frame(val_frame, tokenizer, config)

	train_loader = create_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
	val_loader = create_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
	test_loader = None
	if test_dataset is not None:
		test_loader = create_dataloader(test_dataset, batch_size=config.batch_size, shuffle=False)

	model = PhoBertClassifier(model_name=config.model_name, num_labels=config.num_labels)
	model.to(device)
	class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
	criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

	optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	total_steps = len(train_loader) * config.epochs
	warmup_steps = int(total_steps * config.warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)

	best_f1 = -1.0
	best_metrics = {}
	print(f"Using class weights: {list(config.class_weights)}")

	for epoch in range(1, config.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
		val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
		current_f1 = val_metrics["f1"]

		epoch_metrics = {
			"epoch": epoch,
			"train_loss": train_loss,
			"val_loss": val_metrics["loss"],
			"val_accuracy": val_metrics["accuracy"],
			"val_precision": val_metrics["precision"],
			"val_recall": val_metrics["recall"],
			"val_f1": val_metrics["f1"],
		}

		with (config.log_dir / "training_log.jsonl").open("a", encoding="utf-8") as log_file:
			log_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

		if current_f1 > best_f1:
			best_f1 = current_f1
			best_metrics = epoch_metrics
			save_artifacts(model, tokenizer, config.best_model_dir, best_metrics)

		print(
			f"Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | "
			f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f}"
		)

	with config.metrics_path.open("w", encoding="utf-8") as handle:
		json.dump(best_metrics, handle, ensure_ascii=False, indent=2)

	if test_loader is not None:
		test_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
		with (config.log_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
			json.dump(test_metrics, handle, ensure_ascii=False, indent=2)
		print(
			f"Test metrics | loss={test_metrics['loss']:.4f} | acc={test_metrics['accuracy']:.4f} | "
			f"f1={test_metrics['f1']:.4f}"
		)

	print(f"Best model saved to: {config.best_model_dir}")
	print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
	main()

