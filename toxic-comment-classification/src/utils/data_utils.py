"""Generic data loading and column normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.dataset import infer_label


TEXT_COLUMN_ALIASES = ("comment", "text", "content", "sentence", "review")
LABEL_COLUMN_ALIASES = ("toxicity", "label", "target", "class", "y")


def resolve_column_name(frame: pd.DataFrame, preferred_name: str, aliases: tuple[str, ...]) -> str:
	column_lookup = {str(column).strip().lower(): column for column in frame.columns}
	preferred_key = preferred_name.strip().lower()
	if preferred_key in column_lookup:
		return column_lookup[preferred_key]

	for alias in aliases:
		if alias in column_lookup:
			return column_lookup[alias]

	raise ValueError(
		"Could not detect the required columns. Available columns: "
		f"{list(frame.columns)}"
	)


def read_dataset(file_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
	if not file_path.exists():
		raise FileNotFoundError(f"Data file not found: {file_path}")

	if file_path.suffix.lower() in {".csv", ".txt"}:
		frame = pd.read_csv(file_path)
	elif file_path.suffix.lower() in {".xlsx", ".xls"}:
		frame = pd.read_excel(file_path)
	else:
		raise ValueError("Supported file types are CSV, TXT, XLSX and XLS")

	text_column = resolve_column_name(frame, text_col, TEXT_COLUMN_ALIASES)
	label_column = resolve_column_name(frame, label_col, LABEL_COLUMN_ALIASES)

	frame = frame[[text_column, label_column]].dropna().reset_index(drop=True)
	frame = frame.rename(columns={text_column: text_col, label_column: label_col})
	frame[text_col] = frame[text_col].astype(str)
	frame[label_col] = frame[label_col].apply(infer_label)
	return frame
