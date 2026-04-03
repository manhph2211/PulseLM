#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.io import loadmat


# Fixed question bank used to build QA-style supervision.
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
      "cycling",
      "driving",
      "lunch_break",
      "sitting",
      "stairs",
      "table_soccer",
      "unknown_0",
      "walking",
      "working"
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
    "answers": [
      "af",
      "non_af"
    ]
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
    "answers": [
      "af",
      "pac",
      "pvc",
      "sinus_rhythm",
      "svt",
      "vt"
    ]
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
      "elevated",
      "hypertension_stage1",
      "hypertension_stage2",
      "hypertensive_crisis",
      "normal"
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
    "answers": [
      "bradycardia",
      "normal",
      "tachycardia"
    ]
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
    "answers": [
      "high",
      "low",
      "normal"
    ]
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
    "answers": [
      "high",
      "low",
      "normal"
    ]
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
    "answers": [
      "high",
      "low",
      "normal"
    ]
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
    "answers": [
      "bradypnea",
      "normal",
      "tachypnea"
    ]
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
      "mild_5<=ahi<15",
      "moderate_15<=ahi<30",
      "normal_ahi<5",
      "severe_ahi>=30"
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
      "mild_hypoxemia",
      "moderate_hypoxemia",
      "normal",
      "severe_hypoxemia"
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
    "answers": [
      "good_quality",
      "noisy_or_distorted",
      "symmetric_unusual"
    ]
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
    "answers": [
      "amusement",
      "baseline",
      "meditation",
      "stress"
    ]
  }
}

# ---------------------------------- update text qa extraction ---------------------------------
def _safe_to_str(x) -> str:
    """Convert common MATLAB cell and array wrappers into a clean string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)

    if isinstance(x, np.ndarray):
        # object/cell
        if x.dtype == object:
            if x.size == 0:
                return ""
            # Common layouts include array(['text...'], dtype=object)
            # and array([['text...']], dtype=object).
            return _safe_to_str(x.flat[0])
        # char array
        if x.dtype.kind in ("U", "S"):
            try:
                return str(x.squeeze().tolist())
            except Exception:
                return str(x)
        # numeric -> string
        try:
            return str(x.squeeze())
        except Exception:
            return str(x)

    # list/tuple
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        return _safe_to_str(x[0])

    return str(x)


def _extract_text_by_index(mat: Dict[str, Any], idx: int) -> str:
    if "text" not in mat:
        return ""
    t = mat["text"]

    # Common layouts:
    # 1) text is a cell array with shape (N, 1) or (1, N)
    # 2) text is already list-like
    t = np.array(t)
    if t.dtype == object:
        t = t.squeeze()

    # After squeeze, the value may be a vector or a scalar.
    if isinstance(t, np.ndarray) and t.ndim == 0:
        return _safe_to_str(t.item())

    if isinstance(t, np.ndarray) and t.ndim == 1:
        if idx < len(t):
            return _safe_to_str(t[idx])
        return ""

    # Fall back to the first global entry if the array is not indexed per sample.
    return _safe_to_str(t)


def _extract_qa_by_index(mat: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Return the QA dictionary for one sample index.

    Example:
    {"heart_rate_category": {"question": "...", "answer": "normal"}, ...}
    """
    if "qa" not in mat:
        return {}

    q = np.array(mat["qa"])
    if q.dtype == object:
        q = q.squeeze()

    # After squeeze, the value may be a vector or a scalar.
    if isinstance(q, np.ndarray) and q.ndim == 0:
        s = _safe_to_str(q.item()).strip()
    elif isinstance(q, np.ndarray) and q.ndim == 1:
        if idx >= len(q):
            return {}
        s = _safe_to_str(q[idx]).strip()
    else:
        # Generic fallback.
        s = _safe_to_str(q).strip()

    if not s:
        return {}

    # Some files store JSON as a wrapped string such as ['{...}'].
    if not s.startswith("{"):
        m = re.search(r"(\{.*\})", s)
        if m:
            s = m.group(1)

    try:
        obj = json.loads(s)
    except Exception:
        return {}

    # Some files wrap the JSON object in a list.
    if isinstance(obj, list):
        # Keep the first valid dictionary.
        for it in obj:
            if isinstance(it, dict):
                return it
        return {}
    if isinstance(obj, dict):
        return obj
    return {}
# ---------------------------------- end of text qa update ---------------------------------



def _mat_cell_to_py(x: Any) -> Any:
    """Convert MATLAB cell, char, and numeric arrays into Python objects."""
    if isinstance(x, np.ndarray):
        # char array or object array
        if x.dtype == object:
            if x.size == 1:
                return _mat_cell_to_py(x.item())
            return [_mat_cell_to_py(i) for i in x.flatten()]
        # numeric array
        return x
    return x


def _extract_text(mat: Dict[str, Any]) -> str:
    if "text" not in mat:
        return ""
    v = _mat_cell_to_py(mat["text"])
    # Common layouts are ['...'] or '...'.
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
        return v[0]
    if isinstance(v, str):
        return v
    # Generic fallback.
    try:
        return str(v)
    except Exception:
        return ""


def _extract_fs(mat: Dict[str, Any], default_fs: int = 125) -> int:
    # Common sampling-rate field names.
    for k in ["fs", "sampling_rate_hz", "sampling_rate", "sr", "Fs"]:
        if k in mat:
            v = _mat_cell_to_py(mat[k])
            if isinstance(v, np.ndarray):
                try:
                    v = float(np.array(v).squeeze())
                except Exception:
                    continue
            try:
                iv = int(round(float(v)))
                if iv > 0:
                    return iv
            except Exception:
                pass
    return default_fs


def _extract_signals(mat: Dict[str, Any]) -> np.ndarray:
    """
    Extract the signal matrix from a MAT file.

    The function tries the most common keys first and normalizes the output to
    shape [N, L].
    """
    cand_keys = ["signal", "ppg", "ppg_signal", "signals"]
    key = None
    for k in cand_keys:
        if k in mat:
            key = k
            break
    if key is None:
        raise KeyError(f"Cannot find signal key in mat. tried={cand_keys}, keys={list(mat.keys())[:50]}")

    sig = mat[key]
    sig = np.array(sig)

    # MATLAB may wrap cells in object arrays.
    if sig.dtype == object:
        sig = _mat_cell_to_py(sig)
        sig = np.array(sig)

    # squeeze
    sig = np.array(sig).squeeze()

    # Expand a single 1D signal to [1, L].
    if sig.ndim == 1:
        sig = sig[None, :]

    # Some exports store signals as [L, N] instead of [N, L].
    if sig.ndim == 2 and sig.shape[0] < sig.shape[1]:
        # Keep the array as-is unless the layout strongly suggests a transpose.
        pass
    if sig.ndim == 2 and sig.shape[0] > 0 and sig.shape[1] > 0:
        # Transpose when the first dimension looks like sequence length.
        if sig.shape[0] > 200 and sig.shape[1] < 200:
            sig = sig.T

    if sig.ndim != 2:
        raise ValueError(f"Signal array must be 2D after processing, got shape={sig.shape}")

    return sig.astype(np.float32)


def _extract_qa_list(mat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract QA metadata from a MAT file.

    The raw value may be a wrapped JSON string or a JSON object. The function
    returns a list of dictionaries such as
    {category: {"question": "...", "answer": "..."}, ...}.
    """
    if "qa" not in mat:
        return []

    qa_raw = _mat_cell_to_py(mat["qa"])

    # Common layouts are ['{...}'] or '{...}'.
    json_str = None
    if isinstance(qa_raw, list):
        # The QA field may be list[str] or list[list[str]].
        if len(qa_raw) == 1 and isinstance(qa_raw[0], str):
            json_str = qa_raw[0]
        elif len(qa_raw) == 1 and isinstance(qa_raw[0], list) and len(qa_raw[0]) == 1 and isinstance(qa_raw[0][0], str):
            json_str = qa_raw[0][0]
    elif isinstance(qa_raw, str):
        json_str = qa_raw

    if json_str is None:
        # Fall back to the raw string representation of the array.
        try:
            json_str = str(qa_raw)
        except Exception:
            return []

    json_str = json_str.strip()

    # The decoded QA payload may be either a list or a dictionary.
    try:
        obj = json.loads(json_str)
    except Exception:
        # Some files store JSON deeper inside a string wrapper.
        m = re.search(r"(\{.*\})", json_str)
        if not m:
            return []
        obj = json.loads(m.group(1))

    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        # list[dict]
        return [o for o in obj if isinstance(o, dict)]
    return []


def _make_messages(text_ctx: str, category: str, answer: str, rng: random.Random) -> Tuple[List[Dict[str, str]], str]:
    """Build the chat-style training messages for one example."""
    bank = CATEGORY_BANK[category]
    q = rng.choice(bank["questions"])
    opts = bank["answers"]

    system = (
        "You are a PPG QA classification expert.\n"
        "Rules:\n"
        "- Answer MUST be exactly one option from the provided list.\n"
        "- Output format MUST be strict: <answer>OPTION</answer>\n"
        "- Do not output any extra text.\n"
        "- If the context is insufficient, still choose the best option from the list.\n"
    )

    user = (
        f"Context:\n{text_ctx.strip() if text_ctx.strip() else '(no context provided)'}\n\n"
        f"Task:\n{q}\n\n"
        f"Options:\n" + "\n".join([f"- {o}" for o in opts]) + "\n\n"
        "Return ONLY:\n<answer>OPTION</answer>"
    )

    assistant = f"<answer>{answer}</answer>"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ], q


def _normalize_dataset_name(path: str) -> str:
    base = os.path.basename(path)
    dataset_name = re.sub(r"\.mat$", "", base)
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", dataset_name)


def _split_examples(
    examples: List[Dict[str, Any]],
    split_ratio: Tuple[int, int, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    a, b, c = split_ratio
    total = len(shuffled)
    n_train = int(total * (a / (a + b + c)))
    n_dev = int(total * (b / (a + b + c)))
    n_test = total - n_train - n_dev

    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test


def _write_jsonl(path: str, rows: List[Dict[str, Any]], split_name: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            record["meta"] = dict(record["meta"])
            record["meta"]["split"] = split_name
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_examples_from_mat(
    mat_path: str,
    default_fs: int,
    seed: int,
    signal_ref_path: str = "signals.npy",
    signal_index_offset: int = 0,
) -> Tuple[str, np.ndarray, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    dataset_name = _normalize_dataset_name(mat_path)
    mat = loadmat(mat_path)

    signals = _extract_signals(mat)
    fs = _extract_fs(mat, default_fs=default_fs)
    num_signals, signal_length = signals.shape
    duration_sec = float(signal_length) / float(fs)

    examples: List[Dict[str, Any]] = []
    for idx in range(num_signals):
        text_ctx = _extract_text_by_index(mat, idx) or "(no text provided)"
        qa_obj = _extract_qa_by_index(mat, idx)
        if not qa_obj:
            continue

        for category, payload in qa_obj.items():
            if category not in CATEGORY_BANK or not isinstance(payload, dict):
                continue

            answer = payload.get("answer", None)
            if answer is None:
                continue
            answer = str(answer).strip()
            if answer not in CATEGORY_BANK[category]["answers"]:
                continue

            messages, _ = _make_messages(text_ctx, category, answer, rng)
            example_id = f"ppg_text_qa__{dataset_name}__idx{idx}__{category}__{len(examples)}"
            examples.append(
                {
                    "id": example_id,
                    "meta": {
                        "dataset": dataset_name,
                        "sampling_rate_hz": fs,
                        "duration_sec": duration_sec,
                        "length": signal_length,
                        "signal_storage": "npy_2dp_float32",
                        "question_category": category,
                    },
                    "signal_ref": {
                        "path": signal_ref_path,
                        "index": idx + signal_index_offset,
                    },
                    "label": {
                        "category": category,
                        "answer": answer,
                    },
                    "messages": messages,
                }
            )

    return dataset_name, signals, examples


def build_one_mat(
    mat_path: str,
    out_root: str,
    split_ratio: Tuple[int, int, int] = (8, 1, 1),
    seed: int = 42,
    default_fs: int = 125,
) -> None:
    dataset_name, signals, examples = collect_examples_from_mat(
        mat_path=mat_path,
        default_fs=default_fs,
        seed=seed,
        signal_ref_path="signals.npy",
        signal_index_offset=0,
    )
    if not examples:
        print(f"[SKIP] {mat_path} produced 0 examples after filtering")
        return

    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "signals.npy"), signals)

    train_rows, dev_rows, test_rows = _split_examples(examples, split_ratio, seed)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows, "train")
    _write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev_rows, "dev")
    _write_jsonl(os.path.join(out_dir, "test.jsonl"), test_rows, "test")

    print(
        f"[OK] {dataset_name}: signals={signals.shape} examples={len(examples)} "
        f"train/dev/test={len(train_rows)}/{len(dev_rows)}/{len(test_rows)} -> {out_dir}"
    )


def build_merged_dataset(
    mat_paths: List[str],
    out_dir: str,
    split_ratio: Tuple[int, int, int],
    seed: int,
    default_fs: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    merged_signals: List[np.ndarray] = []
    merged_examples: List[Dict[str, Any]] = []
    signal_offset = 0

    for mat_path in mat_paths:
        dataset_name, signals, examples = collect_examples_from_mat(
            mat_path=mat_path,
            default_fs=default_fs,
            seed=seed,
            signal_ref_path="signals.npy",
            signal_index_offset=signal_offset,
        )
        if not examples:
            print(f"[SKIP] {mat_path} produced 0 examples after filtering")
            continue
        merged_signals.append(signals)
        merged_examples.extend(examples)
        signal_offset += signals.shape[0]
        print(
            f"[MERGE] added {dataset_name}: signals={signals.shape} examples={len(examples)}"
        )

    if not merged_examples:
        raise RuntimeError("No merged examples were produced from the selected MAT files.")

    all_signals = np.concatenate(merged_signals, axis=0)
    np.save(os.path.join(out_dir, "signals.npy"), all_signals)

    train_rows, dev_rows, test_rows = _split_examples(merged_examples, split_ratio, seed)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows, "train")
    _write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev_rows, "dev")
    _write_jsonl(os.path.join(out_dir, "test.jsonl"), test_rows, "test")

    print(
        f"[OK] merged dataset: signals={all_signals.shape} examples={len(merged_examples)} "
        f"train/dev/test={len(train_rows)}/{len(dev_rows)}/{len(test_rows)} -> {out_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing MAT files.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--pattern", type=str, default="*ppg_text.mat")
    ap.add_argument(
        "--mode",
        type=str,
        default="separate",
        choices=["separate", "merged", "both"],
        help="Whether to build one dataset per MAT file, one merged dataset, or both.",
    )
    ap.add_argument(
        "--merged_name",
        type=str,
        default="merged_ppg_text",
        help="Directory name used when --mode includes merged.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--default_fs", type=int, default=125)
    ap.add_argument("--split", type=str, default="8,1,1", help="Train/dev/test ratio, for example 8,1,1.")
    args = ap.parse_args()

    split_ratio = tuple(int(x) for x in args.split.split(","))
    assert len(split_ratio) == 3 and all(x > 0 for x in split_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    mats = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not mats:
        raise SystemExit(f"No MAT file matched {os.path.join(args.in_dir, args.pattern)}")

    if args.mode in ("separate", "both"):
        for mat_path in mats:
            try:
                build_one_mat(
                    mat_path=mat_path,
                    out_root=args.out_dir,
                    split_ratio=split_ratio,
                    seed=args.seed,
                    default_fs=args.default_fs,
                )
            except Exception as exc:
                print(f"[FAIL] {mat_path}: {exc!r}")

    if args.mode in ("merged", "both"):
        try:
            build_merged_dataset(
                mat_paths=mats,
                out_dir=os.path.join(args.out_dir, args.merged_name),
                split_ratio=split_ratio,
                seed=args.seed,
                default_fs=args.default_fs,
            )
        except Exception as exc:
            print(f"[FAIL] merged dataset: {exc!r}")


if __name__ == "__main__":
    main()
