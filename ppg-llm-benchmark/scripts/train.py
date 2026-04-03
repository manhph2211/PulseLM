#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT training for PPG QA-style datasets (JSONL + signals.npy memmap).

This version is adapted for newer JSONL samples like:

{
  "id": "...",
  "meta": {"dataset": "...", "question_category": "...", "split": "train|dev|test", ...},
  "signal_ref": {"path": "signals.npy", "index": 618},
  "messages": [..., {"role":"assistant","content":"<answer>...</answer>"}]
}

Key changes vs previous file:
- No hardcoded HF token (uses env HF_TOKEN if needed)
- Can train from a dataset root directory (auto-picks train/dev jsonl and signals.npy)
- Optionally filters by meta.split if you pass a single mixed jsonl
- Resolves signal_ref.path relative to data_dir (so per-sample signals.npy also works)
- Adds CLI args for common knobs (paths, model, LoRA, batch/epochs, max_length)
"""

import os
import json
import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# -------------------------
# Dataset (JSONL + NPY memmap)
# -------------------------
class PPGJsonlDataset(Dataset):
    """
    Expects JSONL lines with at least:
      - messages (list[{"role":..., "content":...}])
      - signal_ref.index (int)
      - signal_ref.path (optional, default to provided signals_npy_path)
      - meta.split (optional; can be used for filtering)

    Signals:
      - signals.npy: float32 array (N, T) in memmap mode
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Any,
        data_dir: str,
        signals_npy_path: Optional[str] = None,
        max_length: int = 2048,
        ignore_index: int = -100,
        use_chat_template: bool = True,
        split_filter: Optional[str] = None,  # "train" | "dev" | "test"
    ):
        self.jsonl_path = jsonl_path
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.use_chat_template = use_chat_template
        self.split_filter = split_filter

        # Default signals.npy (common case: one shared signals.npy for all samples)
        self.default_signals_path = signals_npy_path or os.path.join(data_dir, "signals.npy")

        # Pre-compute offsets (and optional filtering indices) for fast random access
        self._offsets: List[int] = []
        self._kept_indices: Optional[List[int]] = [] if split_filter else None

        with open(jsonl_path, "rb") as f:
            offset = 0
            i = 0
            for line in f:
                self._offsets.append(offset)
                if split_filter:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if obj.get("meta", {}).get("split") == split_filter:
                            self._kept_indices.append(i)
                    except Exception:
                        # If a line is malformed, keep behavior strict later in __getitem__
                        pass
                offset += len(line)
                i += 1

        # Cache loaded memmaps by absolute signals path (so per-sample signal_ref.path works)
        self._signals_cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        if self._kept_indices is not None:
            return len(self._kept_indices)
        return len(self._offsets)

    def _logical_to_physical_idx(self, idx: int) -> int:
        if self._kept_indices is None:
            return idx
        return self._kept_indices[idx]

    def _read_line(self, physical_idx: int) -> Dict[str, Any]:
        with open(self.jsonl_path, "rb") as f:
            f.seek(self._offsets[physical_idx])
            line = f.readline().decode("utf-8")
        return json.loads(line)

    def _get_signals_memmap(self, signals_path: str) -> np.ndarray:
        abs_path = os.path.abspath(signals_path)
        if abs_path not in self._signals_cache:
            self._signals_cache[abs_path] = np.load(abs_path, mmap_mode="r")
        return self._signals_cache[abs_path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        physical_idx = self._logical_to_physical_idx(idx)
        item = self._read_line(physical_idx)

        # 1) load PPG by index
        sig_idx = int(item["signal_ref"]["index"])
        sr_path = item.get("signal_ref", {}).get("path", None)

        if sr_path:
            # allow relative paths inside JSONL, resolve from data_dir
            signals_path = sr_path if os.path.isabs(sr_path) else os.path.join(self.data_dir, sr_path)
        else:
            signals_path = self.default_signals_path

        signals = self._get_signals_memmap(signals_path)
        ppg = torch.tensor(signals[sig_idx], dtype=torch.float32)  # (T,)

        # 2) locate last assistant message for supervision
        msgs = item["messages"]
        assistant_pos = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "assistant":
                assistant_pos = i
                break
        if assistant_pos is None:
            raise ValueError(f"Sample {item.get('id', physical_idx)} has no assistant message")

        assistant_text = (msgs[assistant_pos].get("content") or "").strip()
        prompt_msgs = msgs[:assistant_pos]

        # 3) build prompt_only + full_text
        use_chat = bool(self.use_chat_template) and hasattr(self.tokenizer, "apply_chat_template")
        if use_chat:
            prompt_only = self.tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                prompt_msgs + [{"role": "assistant", "content": assistant_text}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # fallback: concatenate message content (works if your messages already contain role separators)
            prompt_only = "\n\n".join((m.get("content") or "").strip() for m in prompt_msgs).strip()
            if prompt_only:
                prompt_only = prompt_only + "\n\n"
            full_text = prompt_only + assistant_text

        # 4) tokenize
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt_only,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None or len(attention_mask) != len(input_ids):
            attention_mask = [1] * len(input_ids)

        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        # 5) labels: mask prompt, supervise assistant tokens only
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = self.ignore_index

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ppg": ppg,  # (T,)
        }


# -------------------------
# Data Collator (pad + stack)
# -------------------------
@dataclass
class PPGDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8
    ignore_index: int = -100
    ppg_unsqueeze_channel: bool = False  # True if your ppg_encoder expects (B,1,T)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        max_len = batch["input_ids"].size(1)

        labels = []
        for f in features:
            l = f["labels"]
            if l.size(0) < max_len:
                pad = torch.full((max_len - l.size(0),), self.ignore_index, dtype=torch.long)
                l = torch.cat([l, pad], dim=0)
            else:
                l = l[:max_len]
            labels.append(l)
        batch["labels"] = torch.stack(labels, dim=0)

        ppg = torch.stack([f["ppg"] for f in features], dim=0)  # (B,T)
        if self.ppg_unsqueeze_channel:
            ppg = ppg.unsqueeze(1)  # (B,1,T)
        batch["ppg"] = ppg
        return batch


# -------------------------
# Utilities
# -------------------------
def resolve_default_paths(data_dir: str) -> Tuple[str, str, str]:
    """
    Supports two layouts:
    A) data_dir/train.jsonl, data_dir/dev.jsonl, data_dir/signals.npy
    B) data_dir/*.jsonl (single mixed file) + data_dir/signals.npy  (then use --split_filter)
    """
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    dev_jsonl = os.path.join(data_dir, "dev.jsonl")
    signals_npy = os.path.join(data_dir, "signals.npy")
    return train_jsonl, dev_jsonl, signals_npy


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root containing train/dev jsonl and signals.npy")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save checkpoints/adapters")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--train_jsonl", type=str, default="", help="Optional override. If empty, uses data_dir/train.jsonl")
    parser.add_argument("--dev_jsonl", type=str, default="", help="Optional override. If empty, uses data_dir/dev.jsonl")
    parser.add_argument("--signals_npy", type=str, default="", help="Optional override. If empty, uses data_dir/signals.npy")
    parser.add_argument("--split_filter_train", type=str, default="", help="If train_jsonl is a mixed file, filter by meta.split=train")
    parser.add_argument("--split_filter_dev", type=str, default="", help="If dev_jsonl is a mixed file, filter by meta.split=dev")

    # training knobs
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ppg_proj_lr", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)

    # LoRA knobs
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # multimodal knobs
    parser.add_argument("--ppg_feat_dim", type=int, default=512)
    parser.add_argument("--freeze_ppg_encoder", action="store_true", default=True)
    parser.add_argument("--pulseppg_ckpt", type=str, required=True, help="PulsePPG checkpoint_best.pkl path")
    parser.add_argument("--ppg_unsqueeze_channel", action="store_true", default=False, help="If encoder expects (B,1,T)")

    args = parser.parse_args()

    from transformers import AutoTokenizer, Trainer, TrainingArguments

    from ppg_llm_release.model import MultimodalBPQALLM
    from ppg_llm_release.pulseppg import load_pulseppg_from_checkpoint

    data_dir = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # -------- paths --------
    default_train, default_dev, default_signals = resolve_default_paths(data_dir)
    train_jsonl = args.train_jsonl or default_train
    dev_jsonl = args.dev_jsonl or default_dev
    signals_npy = args.signals_npy or default_signals

    # -------- model / tokenizer --------
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_name,
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- ppg encoder --------
    ppg_encoder = load_pulseppg_from_checkpoint(
        checkpoint_path=args.pulseppg_ckpt,
        device="cpu",
    )

    # Dummy ECG (not used)
    import torch.nn as nn

    class DummyECG(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 128, device=x.device)

    ecg_encoder = DummyECG()

    # -------- multimodal model with LoRA --------
    model = MultimodalBPQALLM(
        llm_name=args.llm_name,
        ecg_encoder=ecg_encoder,
        ppg_encoder=ppg_encoder,
        ppg_feat_dim=args.ppg_feat_dim,
        setting="lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_ecg_encoder=True,
        freeze_ppg_encoder=args.freeze_ppg_encoder,
        token=hf_token,
    )
    
    

    # -------- one-time grad check (rank0) --------
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print("===== TRAINABLE PARAMS (PPG / LoRA) =====")
        for n, p in model.named_parameters():
            nl = n.lower()
            if p.requires_grad and (("lora" in nl) or ("ppg" in nl) or ("proj" in nl)):
                print(f"{n:80s} | requires_grad = {p.requires_grad}")
        print("========================================")

    # -------- dataset / collator --------
    train_split = args.split_filter_train.strip() or None
    dev_split = args.split_filter_dev.strip() or None

    train_ds = PPGJsonlDataset(
        jsonl_path=train_jsonl,
        signals_npy_path=signals_npy,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_chat_template=True,
        split_filter=train_split,
        data_dir=data_dir,
    )
    dev_ds = PPGJsonlDataset(
        jsonl_path=dev_jsonl,
        signals_npy_path=signals_npy,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_chat_template=True,
        split_filter=dev_split,
        data_dir=data_dir,
    )

    collator = PPGDataCollator(
        tokenizer=tokenizer,
        ppg_unsqueeze_channel=args.ppg_unsqueeze_channel,
    )

    # -------- training args --------
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=0.0,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,  # critical: keep ppg in batch

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,

    )

    # -------- custom Trainer: pass ppg to model.forward & separate LR for ppg_proj --------
    class PPGTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            model_ = self.model
            lora_params = []
            ppg_proj_params = []

            for name, param in model_.named_parameters():
                if not param.requires_grad:
                    continue
                lname = name.lower()
                if "ppg_proj" in lname:
                    ppg_proj_params.append(param)
                else:
                    lora_params.append(param)

            if len(ppg_proj_params) == 0:
                print("[WARN] No ppg_proj params found! All trainables will use base LR.")
                param_groups = [{"params": lora_params, "lr": train_args.learning_rate}]
            else:
                param_groups = [
                    {"params": lora_params, "lr": train_args.learning_rate},
                    {"params": ppg_proj_params, "lr": args.ppg_proj_lr},
                ]

            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )

            if int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0):
                for i, g in enumerate(self.optimizer.param_groups):
                    print(f"[OPT GROUP {i}] lr = {g['lr']}, num_params = {len(g['params'])}")
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            ppg = inputs.pop("ppg")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=labels,
                ecg=None,
                ppg=ppg,
            )
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = PPGTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save: adapter + full state (includes ppg_proj)
    trainer.save_model(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "multimodal_full_state.pt"))
    print("Training finished. Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
