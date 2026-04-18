import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, get_dataset_config_names

# Per-category "normal/baseline" answers — everything else is a minority class
# eligible for augmentation. None means no concept of normal (skip augmentation).
_CATEGORY_NORMAL: Dict[str, Optional[Set[str]]] = {
    "activity_label":          None,               # pure activity recognition, no pathological axis
    "af_label":                {"non_af"},
    "arrhythmia_category":     {"sinus_rhythm"},
    "blood_pressure_category": {"normal"},
    "heart_rate_category":     {"normal"},
    "hrv_pnn50_category":      {"normal"},
    "hrv_rmssd_category":      {"normal"},
    "hrv_sdnn_category":       {"normal"},
    "rr_category":             {"normal"},
    "sdb_label":               {"normal_ahi<5"},
    "spo2_category":           {"normal"},
    "sqi_category":            {"good_quality"},
    "stress_label":            {"baseline"},
}


def _is_minority(category: str, answer: str) -> bool:
    """Return True if this (category, answer) should be augmented."""
    normal = _CATEGORY_NORMAL.get(category)
    if normal is None:
        return False          # activity_label — skip augmentation entirely
    return answer not in normal


# ---------------------------------------------------------------------------
# PPG augmentation — label-preserving only (no time warping)
# ---------------------------------------------------------------------------

def _aug_gaussian_noise(sig: np.ndarray) -> np.ndarray:
    std = np.random.uniform(0.005, 0.02)
    return sig + np.random.normal(0.0, std, size=sig.shape).astype(np.float32)


def _aug_amplitude_scale(sig: np.ndarray) -> np.ndarray:
    scale = np.random.uniform(0.90, 1.10)
    return (sig * scale).astype(np.float32)


def _aug_baseline_wander(sig: np.ndarray, fs: float = 125.0) -> np.ndarray:
    freq = np.random.uniform(0.05, 0.4)
    amplitude = np.random.uniform(0.005, 0.03)
    t = np.arange(len(sig), dtype=np.float32) / fs
    return (sig + amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def augment_ppg(sig: np.ndarray) -> np.ndarray:
    """Apply 1–2 random label-preserving augmentations."""
    ops = [_aug_gaussian_noise, _aug_amplitude_scale, _aug_baseline_wander]
    for op in random.sample(ops, k=random.randint(1, 2)):
        sig = op(sig)
    return sig

_ENCODER_HZ = {"papagei": 125, "pulseppg": 50}
_DATASET_HZ = 125  


def _resample_ppg(sig: np.ndarray, encoder_type: str) -> np.ndarray:
    """Resample signal to the encoder's expected sample rate if needed."""
    target_hz = _ENCODER_HZ.get(encoder_type, _DATASET_HZ)
    if target_hz == _DATASET_HZ:
        return sig
    from scipy.signal import resample
    target_len = int(round(len(sig) * target_hz / _DATASET_HZ))
    return resample(sig, target_len).astype(np.float32)

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "category_schema.json")


def load_category_schema() -> Dict[str, Dict[str, List[str]]]:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Loaded once at import time — add new categories to category_schema.json, not here
CATEGORY_SCHEMA: Dict[str, Dict[str, List[str]]] = load_category_schema()

# Maps old HF dataset answer strings to merged/simplified labels.
# Add entries here whenever you collapse classes — no other file needs changing.
ANSWER_REMAP: Dict[str, Dict[str, str]] = {
    "sqi_category": {
        "good_quality": "good_quality",
        "noisy_or_distorted": "noisy",
        "symmetric_unusual": "noisy",
    },
    "spo2_category": {
        "normal": "normal",
        "mild_hypoxemia": "abnormal",
        "moderate_hypoxemia": "abnormal",
        "severe_hypoxemia": "abnormal",
    },
}


def _make_messages(
    category: str,
    answer: str,
    question: str,
) -> List[Dict[str, str]]:
    opts = CATEGORY_SCHEMA[category]["answers"]
    system = (
        "You are a physiological signal analysis expert specializing in PPG-based clinical classification.\n"
        "Rules:\n"
        "- Answer MUST be exactly one option from the provided list.\n"
        "- Output format MUST be strict: <answer>OPTION</answer>\n"
        "- Do not output any extra text.\n"
    )
    user = (
        f"Task:\n{question}\n\n"
        "Options:\n" + "\n".join(f"- {o}" for o in opts) + "\n\n"
        "Return ONLY:\n<answer>OPTION</answer>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"<answer>{answer}</answer>"},
    ]


def _parse_qa(qa: Any) -> Dict[str, Any]:
    if isinstance(qa, dict):
        return qa
    if isinstance(qa, str):
        try:
            return json.loads(qa)
        except Exception:
            return {}
    return {}


class HFPulseLMDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        split: str = "train",
        dataset_names: Optional[List[str]] = None,
        max_length: int = 768,
        ignore_index: int = -100,
        use_chat_template: bool = True,
        seed: int = 42,
        ppg_encoder_type: str = "papagei",
        augment: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.use_chat_template = use_chat_template
        self.ppg_encoder_type = ppg_encoder_type
        self.augment = augment

        if dataset_names is None:
            dataset_names = get_dataset_config_names("Manhph2211/PulseLM")

        self._examples: List[Tuple[np.ndarray, List[Dict[str, str]]]] = []
        self._labels: List[Tuple[str, str]] = []  # (category, answer) per example
        skipped = 0

        for name in dataset_names:
            ds = load_dataset("Manhph2211/PulseLM", name, split=split)
            for row in ds:
                sig = _resample_ppg(np.array(row["signal"], dtype=np.float32), self.ppg_encoder_type)
                qa = _parse_qa(row["qa"])
                if not isinstance(qa, dict):
                    continue
                for category, payload in qa.items():
                    if category == "activity_label":
                        continue
                    if category not in CATEGORY_SCHEMA:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    ans = payload.get("answer", "")
                    ans = ANSWER_REMAP.get(category, {}).get(ans, ans)
                    if ans not in CATEGORY_SCHEMA[category]["answers"]:
                        continue
                    question = payload.get("question")
                    if not question:
                        skipped += 1
                        continue
                    messages = _make_messages(category, ans, question)
                    self._examples.append((sig, messages))
                    self._labels.append((category, ans))

        if skipped:
            print(f"[HFPulseLMDataset] Skipped {skipped} examples missing 'question' field in qa payload.")

    def get_sample_weights(self) -> torch.Tensor:
        """Return per-example weights for WeightedRandomSampler (inverse class frequency).

        Weight is computed per (category, answer) bucket so that every answer
        option within every category gets equal representation.
        """
        from collections import Counter
        counts = Counter(self._labels)
        weights = [1.0 / counts[lbl] for lbl in self._labels]
        t = torch.tensor(weights, dtype=torch.float64)
        # normalize so the mean weight == 1 (keeps effective batch size stable)
        t = t / t.mean()
        return t.float()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sig, msgs = self._examples[idx]
        if self.augment:
            category, ans = self._labels[idx]
            if _is_minority(category, ans):
                sig = augment_ppg(sig)
        ppg = torch.tensor(sig, dtype=torch.float32)

        assistant_pos = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "assistant":
                assistant_pos = i
                break
        if assistant_pos is None:
            raise ValueError(f"Example {idx} has no assistant message")

        assistant_text = (msgs[assistant_pos].get("content") or "").strip()
        prompt_msgs = msgs[:assistant_pos]

        use_chat = self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template")
        if use_chat:
            prompt_only = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = self.tokenizer.apply_chat_template(
                prompt_msgs + [{"role": "assistant", "content": assistant_text}],
                tokenize=False, add_generation_prompt=False,
            )
        else:
            prompt_only = "\n\n".join((m.get("content") or "").strip() for m in prompt_msgs).strip()
            if prompt_only:
                prompt_only = prompt_only + "\n\n"
            full_text = prompt_only + assistant_text

        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors=None)
        prompt_enc = self.tokenizer(prompt_only, truncation=True, max_length=self.max_length, return_tensors=None)

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask") or [1] * len(input_ids)
        if len(attention_mask) != len(input_ids):
            attention_mask = [1] * len(input_ids)

        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = self.ignore_index

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ppg": ppg,
        }


@dataclass
class PPGDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8
    ignore_index: int = -100
    ppg_unsqueeze_channel: bool = False

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

        ppg = torch.stack([f["ppg"] for f in features], dim=0)
        if self.ppg_unsqueeze_channel:
            ppg = ppg.unsqueeze(1)
        batch["ppg"] = ppg
        return batch