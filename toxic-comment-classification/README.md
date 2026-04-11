# Toxic Comment Classification with PhoBERT

This project trains a binary classifier to separate toxic and non-toxic comments using PhoBERT.

## Expected Data Format

Use a CSV, TXT, XLS, or XLSX file with at least two columns.

This project will auto-detect common names like:

- text column: `Comment`, `text`, `content`
- label column: `Toxicity`, `label`, `target`

The code is case-insensitive, so `Comment` and `comment` both work.

Example expected schema:

- `Comment`: the comment content
- `Toxicity`: the target class, usually `0` for non-toxic and `1` for toxic

Example:

```csv
Comment,Toxicity
"Bạn thật tuyệt vời",0
"Cút đi",1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train

Run from the repository root:

```bash
python -m src.training.train \
	--train-file data/raw/ViCTSD_train.csv \
	--valid-file data/raw/ViCTSD_valid.csv \
	--test-file data/raw/ViCTSD_test.csv \
	--text-col Comment \
	--label-col Toxicity
```

You can still run single-file mode (auto split train/val):

```bash
python -m src.training.train --data-file data/raw/train.csv --text-col Comment --label-col Toxicity
```

Useful options:

- `--model-name vinai/phobert-base`
- `--batch-size 16`
- `--max-length 256`
- `--epochs 4`
- `--output-dir outputs`

## Output

Training will create:

- `outputs/checkpoints/best_model/` for the best model and tokenizer
- `outputs/logs/training_log.jsonl` for per-epoch metrics
- `outputs/logs/metrics.json` for the best validation metrics
- `outputs/logs/test_metrics.json` when `--test-file` is provided

## Notes

- The text cleaner is intentionally light so PhoBERT can keep most of the original signal.
- If your dataset uses different column names, pass them with `--text-col` and `--label-col`.
