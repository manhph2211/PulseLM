# PPG LLM Benchmark

This folder contains a compact public-facing version of the PPG-to-LLM training pipeline.

Recommended Python version: 3.10 or 3.11.

## What is included

- `scripts/prepare_dataset.py`
  - Converts raw `.mat` files into JSONL + `signals.npy`.
  - Supports one dataset per source file, one merged dataset, or both.
- `scripts/train.py`
  - Trains the multimodal model with LoRA.
  - Uses `meta-llama/Llama-3.1-8B-Instruct` as the example base model.
- `scripts/test.py`
  - Runs inference on a prepared dataset and writes `eval_predictions.jsonl` and `eval_metrics.json`.
- `scripts/evaluate_predictions.py`
  - Re-computes aggregate metrics directly from `eval_predictions.jsonl`.
- `src/ppg_llm_release/`
  - Minimal model code plus bundled PPG and ECG encoder source files.
- `examples/*.sh`
  - Fixed example commands for dataset preparation, training, testing, and metric aggregation.

## Dataset format

Each prepared dataset directory contains:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `signals.npy`

Each JSONL record stores:

- metadata such as dataset name, split, question category, sampling rate, and signal length
- a `signal_ref` pointing to `signals.npy`
- a `label.answer`
- a `messages` list formatted for chat-style supervised fine-tuning

## Installation

```bash
cd ppg-llm-benchmark
pip install -r requirements.txt
```

## Example shell scripts

The `examples/` directory contains ready-to-run shell scripts:

- `examples/prepare_dataset_example.sh`
- `examples/train_llama31_8b_example.sh`
- `examples/train_ddp_example.sh`
- `examples/test_llama31_8b_example.sh`
- `examples/test_all_datasets_example.sh`
- `examples/evaluate_predictions_example.sh`
- `examples/run_pipeline_example.sh`

Each script uses environment variables with placeholder defaults, so the usual workflow is:

```bash
cd ppg-llm-benchmark
bash examples/train_llama31_8b_example.sh
```

or:

```bash
cd ppg-llm-benchmark
DATA_DIR=/real/data/path \
OUT_DIR=/real/output/path \
PULSEPPG_CKPT=/real/checkpoint_best.pkl \
bash examples/train_llama31_8b_example.sh
```

## 1. Prepare datasets

Build one dataset per `.mat` file:

```bash
python scripts/prepare_dataset.py \
  --in_dir /path/to/raw_mat_files \
  --out_dir /path/to/processed_datasets \
  --mode separate \
  --pattern "*ppg_text.mat"
```

Build one merged dataset:

```bash
python scripts/prepare_dataset.py \
  --in_dir /path/to/raw_mat_files \
  --out_dir /path/to/processed_datasets \
  --mode merged \
  --merged_name merged_ppg_text \
  --pattern "*ppg_text.mat"
```

Build both:

```bash
python scripts/prepare_dataset.py \
  --in_dir /path/to/raw_mat_files \
  --out_dir /path/to/processed_datasets \
  --mode both
```

Equivalent example shell script:

```bash
bash examples/prepare_dataset_example.sh
```

## 2. Train with Llama-3.1-8B-Instruct

```bash
python scripts/train.py \
  --data_dir /path/to/processed_datasets/vitaldb_ppg_text \
  --out_dir /path/to/checkpoints/data4_vitaldb_ppg_text \
  --pulseppg_ckpt /path/to/checkpoint_best.pkl \
  --llm_name meta-llama/Llama-3.1-8B-Instruct \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --max_length 1024
```

Equivalent example shell script:

```bash
bash examples/train_llama31_8b_example.sh
```

DDP example:

```bash
bash examples/train_ddp_example.sh
```

The script saves:

- Hugging Face checkpoint files
- `multimodal_full_state.pt`

## 3. Test

```bash
python scripts/test.py \
  --data_dir /path/to/processed_datasets/afppg_ppg_text \
  --out_dir /path/to/eval/afppg_ppg_text \
  --full_state_path /path/to/checkpoints/data4_vitaldb_ppg_text/multimodal_full_state.pt \
  --pulseppg_ckpt /path/to/checkpoint_best.pkl \
  --llm_name meta-llama/Llama-3.1-8B-Instruct \
  --split_filter test \
  --max_new_tokens 32
```

Equivalent example shell script:

```bash
bash examples/test_llama31_8b_example.sh
```

Run testing on every dataset directory under one root:

```bash
bash examples/test_all_datasets_example.sh
```

The script writes:

- `eval_predictions.jsonl`
- `eval_metrics.json`

## 4. Evaluate saved predictions

```bash
python scripts/evaluate_predictions.py \
  --predictions /path/to/eval/afppg_ppg_text/eval_predictions.jsonl \
  --out_dir /path/to/eval/afppg_ppg_text
```

Equivalent example shell script:

```bash
bash examples/evaluate_predictions_example.sh
```

This produces:

- `prediction_metrics.json`

## 5. Run the full example pipeline

```bash
bash examples/run_pipeline_example.sh
```

## Notes

- In this project, `DDP` means `DistributedDataParallel`. It is a multi-GPU training launch mode driven by `torchrun`. It is not a dataset name.
- The scripts expect a local PulsePPG checkpoint file.
- Access to gated Hugging Face models such as Llama may require `HF_TOKEN`.
- The code is organized for clarity and portability rather than for preserving every historical experiment in the original project.
