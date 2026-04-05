import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, get_dataset_config_names

CATEGORY_BANK: Dict[str, Dict[str, List[str]]] = {
"activity_label": {
    "questions": [
      "Categorize the physical activity.",
      "Classify the activity type.",
      "Determine the activity category for this sample.",
      "Identify the physical activity for this segment.",
      "Identify what the subject was doing during this recording.",
      "What activity does this PPG correspond to?",
      "What activity was being performed during this PPG recording?",
      "What is the activity label for this sample?",
      "What is the motion/activity state for this segment?",
      "What type of activity is shown in this recording?"
    ],
    "answers": [
      "cycling", "driving", "lunch_break", "sitting", "stairs",
      "table_soccer", "unknown_0", "walking", "working"
    ]
  },
  "af_label": {
    "questions": [
      "Assess whether atrial fibrillation is present in this PPG.",
      "Classify this PPG as AF or non-AF.",
      "Determine whether this signal indicates atrial fibrillation.",
      "Does this PPG signal show atrial fibrillation?",
      "Does this waveform indicate AF?",
      "Is atrial fibrillation present in this recording?",
      "Is this a normal rhythm or atrial fibrillation?",
      "Provide the atrial fibrillation detection result.",
      "What is the AF detection result for this segment?",
      "What is the AF label for this PPG recording?"
    ],
    "answers": ["af", "non_af"]
  },
  "arrhythmia_category": {
    "questions": [
      "Assess the arrhythmia classification from this recording.",
      "Categorize the heart rhythm abnormality.",
      "Classify the cardiac rhythm in this recording.",
      "Determine the rhythm classification for this waveform.",
      "Identify the arrhythmia type from the signal.",
      "Is this a normal rhythm or arrhythmia?",
      "What cardiac rhythm category does this sample belong to?",
      "What is the arrhythmia category for this segment?",
      "What is the rhythm diagnosis for this segment?",
      "What type of arrhythmia does this signal show?"
    ],
    "answers": ["af", "pac", "pvc", "sinus_rhythm", "svt", "vt"]
  },
  "blood_pressure_category": {
    "questions": [
      "Based on the PPG, what is the BP category?",
      "Categorize the blood pressure level.",
      "Classify the blood pressure level shown in this PPG segment.",
      "Determine the BP classification from this waveform.",
      "Does this sample indicate normal blood pressure or hypertension?",
      "Provide the blood pressure risk category.",
      "What blood pressure class does this sample belong to?",
      "What hypertension stage does this PPG correspond to?",
      "What is the blood pressure category for this sample?",
      "What is the blood pressure status for this recording?"
    ],
    "answers": [
      "elevated", "hypertension_stage1", "hypertension_stage2",
      "hypertensive_crisis", "normal"
    ]
  },
  "heart_rate_category": {
    "questions": [
      "Based on the PPG waveform, what is the HR category?",
      "Categorize the heart rate shown in this recording.",
      "Classify the heart rate based on this waveform.",
      "Determine the heart rate category from the signal.",
      "Is the heart rate normal, bradycardic, or tachycardic?",
      "Provide the clinical heart rate category.",
      "What heart rate classification does this PPG indicate?",
      "What is the heart rate category for this PPG segment?",
      "What is the heart rate status for this sample?",
      "Which heart rate class does this sample belong to?"
    ],
    "answers": ["bradycardia", "normal", "tachycardia"]
  },
  "hrv_pnn50_category": {
    "questions": [
      "Assess the pNN50 category from this PPG.",
      "Categorize the pNN50 variability measure.",
      "Classify the pNN50 level.",
      "Determine the pNN50 level for this recording.",
      "How would you categorize pNN50 for this PPG?",
      "Is pNN50 low, normal, or high in this sample?",
      "Provide the pNN50 category.",
      "What is the pNN50 category for this segment?",
      "What is the pNN50-based HRV classification?",
      "What pNN50 class does this sample belong to?"
    ],
    "answers": ["high", "low", "normal"]
  },
  "hrv_rmssd_category": {
    "questions": [
      "Assess the RMSSD-based variability level.",
      "Categorize the short-term HRV (RMSSD).",
      "Classify the RMSSD-based heart rate variability.",
      "Determine the RMSSD classification.",
      "How would you categorize RMSSD here?",
      "Is the RMSSD low, normal, or high?",
      "Provide the RMSSD category for this sample.",
      "What RMSSD class does this recording indicate?",
      "What is the HRV RMSSD category for this segment?",
      "What is the parasympathetic activity level (RMSSD)?"
    ],
    "answers": ["high", "low", "normal"]
  },
  "hrv_sdnn_category": {
    "questions": [
      "Assess the SDNN-based variability category.",
      "Categorize the overall HRV (SDNN) level.",
      "Classify the SDNN-based heart rate variability level.",
      "Determine the SDNN level for this recording.",
      "How would you categorize SDNN for this PPG?",
      "Is the SDNN low, normal, or high in this sample?",
      "Provide the SDNN category based on this PPG segment.",
      "What SDNN class does this sample belong to?",
      "What is the HRV SDNN category for this segment?",
      "What is the SDNN-based HRV classification?"
    ],
    "answers": ["high", "low", "normal"]
  },
  "rr_category": {
    "questions": [
      "Assess the respiratory rate category from this PPG.",
      "Categorize the breathing rate.",
      "Classify the respiratory rate level.",
      "Determine the respiratory rate category for this recording.",
      "How would you categorize the respiratory rate here?",
      "Is the respiratory rate normal, slow, or fast?",
      "Provide the respiratory rate category.",
      "What is the breathing rate classification for this segment?",
      "What is the respiratory rate category for this sample?",
      "What respiratory rate class does this sample belong to?"
    ],
    "answers": ["bradypnea", "normal", "tachypnea"]
  },
  "sdb_label": {
    "questions": [
      "Assess the respiratory disturbance category.",
      "Categorize the breathing disorder level.",
      "Classify the sleep breathing pattern.",
      "Determine the sleep apnea severity.",
      "Does this segment indicate sleep apnea?",
      "Provide the SDB classification for this PPG window.",
      "What is the AHI-based severity category?",
      "What is the breathing disorder category?",
      "What is the sleep-disordered breathing label for this segment?",
      "What sleep-disordered breathing class is this?"
    ],
    "answers": [
      "mild_5<=ahi<15", "moderate_15<=ahi<30", "normal_ahi<5", "severe_ahi>=30"
    ]
  },
  "spo2_category": {
    "questions": [
      "Assess the oxygen saturation level from this PPG.",
      "Categorize the SpO2 level.",
      "Classify the blood oxygen saturation.",
      "Determine the SpO2 category for this recording.",
      "How would you categorize SpO2 for this PPG?",
      "Is the SpO2 normal or does it indicate hypoxemia?",
      "Provide the oxygen saturation category.",
      "What SpO2 class does this sample belong to?",
      "What is the SpO2 category for this segment?",
      "What is the oxygen saturation classification?"
    ],
    "answers": [
      "mild_hypoxemia", "moderate_hypoxemia", "normal", "severe_hypoxemia"
    ]
  },
  "sqi_category": {
    "questions": [
      "Assess the signal quality of this PPG recording.",
      "Classify the PPG signal quality based on skewness.",
      "Determine the signal quality index category.",
      "How would you categorize the signal quality here?",
      "Is this PPG recording of good or poor quality?",
      "Is this PPG signal clean or motion distorted?",
      "Provide the SQI quality category for this sample.",
      "Rate the quality of this PPG signal.",
      "What is the SQI classification for this segment?",
      "What is the signal quality category for this PPG waveform?"
    ],
    "answers": ["good_quality", "noisy_or_distorted", "symmetric_unusual"]
  },
  "stress_label": {
    "questions": [
      "Categorize the stress condition for this sample.",
      "Classify the stress level from this PPG.",
      "Determine the emotional/stress state.",
      "Identify the stress level for this segment.",
      "Provide the stress state for this PPG window.",
      "What is the affective state for this recording?",
      "What is the emotional state label?",
      "What is the stress label for this segment?",
      "What psychological state does this segment indicate?",
      "What stress category does this sample belong to?"
    ],
    "answers": ["amusement", "baseline", "meditation", "stress"]
  }
}


def _make_messages(
    text_ctx: str,
    category: str,
    answer: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    bank = CATEGORY_BANK[category]
    q = rng.choice(bank["questions"])
    opts = bank["answers"]
    system = (
        "You are a physiological signal analysis expert specializing in PPG-based clinical classification.\n"
        "Rules:\n"
        "- Answer MUST be exactly one option from the provided list.\n"
        "- Output format MUST be strict: <answer>OPTION</answer>\n"
        "- Do not output any extra text.\n"
        "- If the context is insufficient, still choose the best option from the list.\n"
    )
    user = (
        f"Context:\n{text_ctx.strip() if text_ctx.strip() else '(no context provided)'}\n\n"
        f"Task:\n{q}\n\n"
        f"Options:\n" + "\n".join(f"- {o}" for o in opts) + "\n\n"
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
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.use_chat_template = use_chat_template
        self._rng = random.Random(seed)

        if dataset_names is None:
            dataset_names = get_dataset_config_names("Manhph2211/PulseLM")

        self._examples: List[Tuple[np.ndarray, List[Dict[str, str]]]] = []

        for name in dataset_names:
            ds = load_dataset("Manhph2211/PulseLM", name, split=split)
            for row in ds:
                sig = np.array(row["signal"], dtype=np.float32)
                text_ctx = (row["text"] or "").strip()
                qa = _parse_qa(row["qa"])
                if not isinstance(qa, dict):
                    continue
                for category, payload in qa.items():
                    if category not in CATEGORY_BANK:
                        continue
                    ans = payload.get("answer", str(payload)) if isinstance(payload, dict) else str(payload)
                    if ans not in CATEGORY_BANK[category]["answers"]:
                        continue
                    messages = _make_messages(text_ctx, category, ans, self._rng)
                    self._examples.append((sig, messages))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sig, msgs = self._examples[idx]
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
