import os
import json
import re
import argparse
import random
import sys
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)


def load_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def build_prompt(messages: List[Dict[str, Any]], tokenizer) -> str:
    prompt_msgs = [m for m in messages if m.get("role") in ("system", "user")]
    return tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)


def get_options(item: Dict[str, Any]) -> List[str]:
    from dataset import CATEGORY_SCHEMA
    qcat = (item.get("meta") or {}).get("question_category", "")
    if qcat in CATEGORY_SCHEMA:
        return CATEGORY_SCHEMA[qcat]["answers"]
    return []


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(?P<ans>.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def _canon_from_options(ans_text: str, options: List[str]) -> Optional[str]:
    if ans_text is None:
        return None
    a = ans_text.strip().lower()
    if not a:
        return None
    for o in options:
        if o.strip().lower() == a:
            return o
    return None


def parse_answer(text: str, options: List[str]) -> Dict[str, Any]:
    out = {
        "answer": None,
        "raw_extracted": None,
        "ok_answer_tag": False,
        "in_options": False,
    }
    if not text:
        return out
    matches = list(_ANSWER_TAG_RE.finditer(text))
    if matches:
        raw = matches[-1].group("ans")
        out["raw_extracted"] = raw.strip()
        canon = _canon_from_options(raw, options)
        if canon is not None:
            out["answer"] = canon
            out["ok_answer_tag"] = True
            out["in_options"] = True
            return out
    opts_sorted = sorted(options, key=lambda x: len(x), reverse=True)
    if not opts_sorted:
        return out
    alts = "|".join(re.escape(o) for o in opts_sorted)
    m2s = list(re.finditer(rf"(?i)\b({alts})\b", text))
    if m2s:
        raw = m2s[-1].group(1)
        out["raw_extracted"] = raw
        canon = _canon_from_options(raw, options)
        if canon is not None:
            out["answer"] = canon
            out["in_options"] = True
    return out


class SignalsCache:
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, base_dir: str, rel_or_abs_path: str) -> np.ndarray:
        p = rel_or_abs_path if os.path.isabs(rel_or_abs_path) else os.path.join(base_dir, rel_or_abs_path)
        if p not in self._cache:
            if not os.path.exists(p):
                raise FileNotFoundError(f"signals file not found: {p}")
            self._cache[p] = np.load(p, mmap_mode="r")
        return self._cache[p]


def load_hf_items(dataset_names, split, seed, ppg_encoder_type="papagei", category_remap=None):
    """Load items from HuggingFace dataset.

    category_remap: optional dict for cross-category evaluation, e.g.:
        {"arrhythmia_category": {"target": "af_label", "answer_map": {"af": "af", "sinus_rhythm": "non_af"}}}
    When a source category appears in category_remap, rows are remapped to the target
    category and answer_map. Rows whose answer is not in answer_map are skipped.
    """
    from datasets import load_dataset, get_dataset_config_names
    from dataset import CATEGORY_SCHEMA, _make_messages, _parse_qa, _resample_ppg

    if dataset_names is None:
        dataset_names = get_dataset_config_names("Manhph2211/PulseLM")
    category_remap = category_remap or {}

    items = []
    skipped = 0
    remapped = 0
    for name in dataset_names:
        ds = load_dataset("Manhph2211/PulseLM", name, split=split)
        for row_idx, row in enumerate(ds):
            sig = _resample_ppg(np.array(row["signal"], dtype=np.float32), ppg_encoder_type)
            qa = _parse_qa(row["qa"])
            if not isinstance(qa, dict):
                continue
            for category, payload in qa.items():
                if not isinstance(payload, dict):
                    continue
                ans = payload.get("answer", "")
                question = payload.get("question")

                # Apply category remap if configured for this source category
                if category in category_remap:
                    remap = category_remap[category]
                    target_cat = remap["target"]
                    answer_map = remap["answer_map"]
                    if ans not in answer_map:
                        continue  # this answer is not in the binary subset — skip
                    mapped_ans = answer_map[ans]
                    if target_cat not in CATEGORY_SCHEMA:
                        continue
                    if mapped_ans not in CATEGORY_SCHEMA[target_cat]["answers"]:
                        continue
                    if not question:
                        skipped += 1
                        continue
                    messages = _make_messages(target_cat, mapped_ans, question)
                    items.append({
                        "id": f"{name}__row{row_idx}__{category}__as__{target_cat}",
                        "meta": {"dataset": name, "question_category": target_cat,
                                 "source_category": category, "split": split},
                        "label": {"answer": mapped_ans},
                        "messages": messages,
                        "_signal_array": sig,
                    })
                    remapped += 1
                    continue

                # Normal path
                if category not in CATEGORY_SCHEMA:
                    continue
                if ans not in CATEGORY_SCHEMA[category]["answers"]:
                    continue
                if not question:
                    skipped += 1
                    continue
                messages = _make_messages(category, ans, question)
                items.append({
                    "id": f"{name}__row{row_idx}__{category}",
                    "meta": {"dataset": name, "question_category": category, "split": split},
                    "label": {"answer": ans},
                    "messages": messages,
                    "_signal_array": sig,
                })
    if skipped:
        print(f"[load_hf_items] Skipped {skipped} examples missing 'question' field in qa payload.")
    if remapped:
        print(f"[load_hf_items] Remapped {remapped} examples via category_remap.")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_hf_dataset", action="store_true", default=False)
    ap.add_argument("--hf_test_names", type=str, default="",
                    help="Comma-separated HF configs to evaluate on. Empty = all.")
    ap.add_argument("--hf_split", type=str, default="test")
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--split_filter", type=str, default="test")
    ap.add_argument("--question_category_filter", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--full_state_path", type=str, default="")
    ap.add_argument("--skip_ckpt", action="store_true", default=False,
                    help="Skip loading checkpoint — runs with random weights to verify the pipeline.")
    ap.add_argument("--ppg_encoder_type", type=str, default="papagei", choices=["pulseppg", "papagei"])
    ap.add_argument("--ppg_encoder_ckpt", type=str, required=True)
    ap.add_argument("--llm_name", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--category_remap", type=str, default=None,
                    help="JSON string for cross-category eval, e.g. "
                         "'{\"arrhythmia_category\": {\"target\": \"af_label\", "
                         "\"answer_map\": {\"af\": \"af\", \"sinus_rhythm\": \"non_af\"}}}'")

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--ppg_unsqueeze_channel", action="store_true")
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--max_print_failures", type=int, default=30)
    ap.add_argument("--ppg_feat_dim", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models.pulselm import MultimodalPPGLLM
    from models.ppg_encoder.pulseppg import load_pulseppg_from_checkpoint
    from models.ppg_encoder.papagei import load_papagei_from_checkpoint

    ensure_dir(args.out_dir)
    result_path = os.path.join(args.out_dir, "eval_metrics.json")
    pred_jsonl_path = os.path.join(args.out_dir, "eval_predictions.jsonl")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    if not args.skip_ckpt and not os.path.exists(args.full_state_path):
        raise FileNotFoundError(f"Missing full_state: {args.full_state_path}")
    if not os.path.exists(args.ppg_encoder_ckpt):
        raise FileNotFoundError(f"Missing PPG encoder ckpt: {args.ppg_encoder_ckpt}")

    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    jsonl_path = None
    sf = None
    category_remap = None
    if args.category_remap:
        try:
            category_remap = json.loads(args.category_remap)
        except json.JSONDecodeError as e:
            raise ValueError(f"--category_remap is not valid JSON: {e}")

    if args.use_hf_dataset:
        hf_test_names = [n.strip() for n in args.hf_test_names.split(",") if n.strip()] or None
        items = load_hf_items(hf_test_names, args.hf_split, args.seed,
                              ppg_encoder_type=args.ppg_encoder_type,
                              category_remap=category_remap)
        num_loaded_raw = len(items)
        print(f"Loaded {num_loaded_raw} items from HF (split={args.hf_split})")
    else:
        jsonl_path = args.jsonl or os.path.join(args.data_dir, "test.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Missing jsonl: {jsonl_path}")
        items = load_items(jsonl_path)
        num_loaded_raw = len(items)
        sf = (args.split_filter or "").strip()
        if sf:
            items = [it for it in items if (it.get("meta", {}) or {}).get("split") == sf]
        print(f"Loaded raw={num_loaded_raw} from {jsonl_path}")
        if sf:
            print(f"Applied split_filter={sf} => remaining={len(items)}")

    qf = (args.question_category_filter or "").strip()
    if qf:
        items = [it for it in items if (it.get("meta", {}) or {}).get("question_category") == qf]
        print(f"Applied question_category_filter={qf} => remaining={len(items)}")

    random.shuffle(items)
    if args.max_samples and args.max_samples > 0:
        items = items[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, cache_dir=args.cache_dir, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.ppg_encoder_type == "pulseppg":
        ppg_encoder = load_pulseppg_from_checkpoint(args.ppg_encoder_ckpt, device="cpu")
    else:
        ppg_encoder = load_papagei_from_checkpoint(args.ppg_encoder_ckpt, device="cpu")

    model = MultimodalPPGLLM(
        llm_name=args.llm_name,
        ppg_encoder=ppg_encoder,
        setting="lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        cache_dir=args.cache_dir,
        token=hf_token,
        freeze_ppg_encoder=True,
        ppg_feat_dim=args.ppg_feat_dim,
    )

    missing, unexpected = [], []
    if args.skip_ckpt:
        print("[skip_ckpt] Running with random weights — pipeline check only, predictions will be meaningless.")
    else:
        state = torch.load(args.full_state_path, map_location="cpu")
        state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        del state
        print(f"Loaded full_state. missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(torch.bfloat16).to(device).eval()

    # Left-padding is required for batched generation so all sequences are right-aligned
    tokenizer.padding_side = "left"

    skip = Counter()
    overall = Counter()
    per_q = defaultdict(Counter)
    per_ds = defaultdict(Counter)
    failures_printed = 0
    signals_cache = SignalsCache()
    fout = open(pred_jsonl_path, "w", encoding="utf-8")

    gen_base_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.temperature is not None and args.do_sample:
        gen_base_kwargs["temperature"] = args.temperature

    items_processed = 0

    for batch_start in range(0, len(items), args.batch_size):
        batch_all = items[batch_start:batch_start + args.batch_size]

        # --- Validate and prepare each item in the batch ---
        ready = []  # (it, ppg_tensor, options, gt_canon, sig_path, sig_idx)
        for it in batch_all:
            meta = it.get("meta", {}) or {}
            qcat = meta.get("question_category", "UNKNOWN")
            dset = meta.get("dataset", "UNKNOWN")
            per_q[qcat]["num_seen"] += 1
            per_ds[dset]["num_seen"] += 1

            options = get_options(it)
            if not options:
                skip["no_options"] += 1
                per_q[qcat]["skip_no_options"] += 1
                continue

            gt = it.get("label", {}) or {}
            gt_ans = gt.get("answer", None)
            if not isinstance(gt_ans, str) or not gt_ans.strip():
                skip["missing_gt"] += 1
                per_q[qcat]["skip_missing_gt"] += 1
                continue

            gt_canon = _canon_from_options(gt_ans, options)
            if gt_canon is None:
                skip["gt_not_in_options"] += 1
                per_q[qcat]["skip_gt_not_in_options"] += 1
                continue

            per_q[qcat]["num_eligible"] += 1

            sig_path = None
            sig_idx = None
            if "_signal_array" in it:
                ppg = torch.tensor(it["_signal_array"], dtype=torch.float32)
            else:
                sig_ref = it.get("signal_ref", {}) or {}
                try:
                    sig_idx = int(sig_ref.get("index"))
                except Exception:
                    skip["bad_signal_index"] += 1
                    per_q[qcat]["skip_bad_signal_index"] += 1
                    continue
                sig_path = sig_ref.get("path", "signals.npy")
                try:
                    arr = signals_cache.get(args.data_dir, sig_path)
                except FileNotFoundError:
                    skip["signals_file_missing"] += 1
                    per_q[qcat]["skip_signals_file_missing"] += 1
                    continue
                if sig_idx < 0 or sig_idx >= len(arr):
                    skip["signal_index_oob"] += 1
                    per_q[qcat]["skip_signal_index_oob"] += 1
                    continue
                ppg = torch.tensor(arr[sig_idx], dtype=torch.float32)

            if args.ppg_unsqueeze_channel:
                ppg = ppg.unsqueeze(0)

            ready.append((it, ppg, options, gt_canon, sig_path, sig_idx))

        if not ready:
            continue

        # --- Batch tokenize (left-padded) ---
        prompts = [build_prompt(it.get("messages", []) or [], tokenizer) for it, *_ in ready]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # --- Batch PPG ---
        ppg_batch = torch.stack([ppg for _, ppg, *_ in ready]).to(device)

        # --- Generate for entire batch ---
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ppg=ppg_batch,
                **gen_base_kwargs,
            )

        # --- Score each item ---
        for j, (it, ppg, options, gt_canon, sig_path, sig_idx) in enumerate(ready):
            items_processed += 1
            meta = it.get("meta", {}) or {}
            qcat = meta.get("question_category", "UNKNOWN")
            dset = meta.get("dataset", "UNKNOWN")
            gt = it.get("label", {}) or {}
            gt_ans = gt.get("answer", "")

            out_text = tokenizer.decode(gen_ids[j], skip_special_tokens=True).strip()

            if out_text == "":
                skip["empty_output"] += 1
                per_q[qcat]["empty_output"] += 1
                continue

            parsed = parse_answer(out_text, options)
            pred_ans = parsed["answer"]

            overall["scored"] += 1
            per_q[qcat]["scored"] += 1
            per_ds[dset]["scored"] += 1

            if parsed["raw_extracted"] is not None:
                overall["parse_ok"] += 1
                per_q[qcat]["parse_ok"] += 1

            if pred_ans is not None and parsed["in_options"]:
                overall["pred_in_options"] += 1
                per_q[qcat]["pred_in_options"] += 1
            else:
                skip["pred_not_in_options_or_parse_failed"] += 1
                per_q[qcat]["skip_pred_not_in_options_or_parse_failed"] += 1

            is_correct = (pred_ans == gt_canon)
            if is_correct:
                overall["correct"] += 1
                per_q[qcat]["correct"] += 1
                per_ds[dset]["correct"] += 1

            record = {
                "id": it.get("id"),
                "meta": meta,
                "signal_ref": {"path": sig_path, "index": sig_idx},
                "gt": {"answer": gt_canon, "raw": gt_ans, "options": options},
                "pred": {"answer": pred_ans},
                "parse": {
                    "ok_answer_tag": parsed["ok_answer_tag"],
                    "raw_extracted": parsed["raw_extracted"],
                    "in_options": parsed["in_options"],
                },
                "raw_text": out_text,
                "is_correct": is_correct,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if failures_printed < args.max_print_failures:
                need_print = False
                reasons = []
                if pred_ans is None:
                    need_print = True
                    reasons.append("parse_failed_or_not_in_options")
                elif not is_correct:
                    need_print = True
                    reasons.append("answer_mismatch")
                if need_print:
                    failures_printed += 1
                    print(f"\n{'='*80}")
                    print(f"[FAIL #{failures_printed}] qcat={qcat} id={it.get('id')} idx={sig_idx} reason={', '.join(reasons)}")
                    print(f"OPTIONS: {options}")
                    print(f"GT:   {gt_canon}")
                    print(f"PRED: {pred_ans}")
                    print(f"RAW_OUT: {out_text}")
                    print(f"{'='*80}\n")

        if (items_processed // args.print_every) != ((items_processed - len(ready)) // args.print_every):
            acc = (overall["correct"] / overall["scored"]) if overall["scored"] else 0.0
            pr = (overall["pred_in_options"] / overall["scored"]) if overall["scored"] else 0.0
            print(f"[PROGRESS] {items_processed}/{len(items)} scored={overall['scored']} acc={acc:.4f} pred_in_options={pr:.4f} skips={sum(skip.values())}")

    fout.close()

    def safe_div(a: int, b: int) -> Optional[float]:
        return (a / b) if b else None

    per_q_metrics = {}
    for qcat, c in per_q.items():
        scored = int(c.get("scored", 0))
        per_q_metrics[qcat] = {
            "num_seen": int(c.get("num_seen", 0)),
            "num_eligible": int(c.get("num_eligible", 0)),
            "num_scored": scored,
            "accuracy": safe_div(int(c.get("correct", 0)), scored),
            "parse_rate": safe_div(int(c.get("parse_ok", 0)), scored),
            "pred_in_options_rate": safe_div(int(c.get("pred_in_options", 0)), scored),
            "empty_output_count": int(c.get("empty_output", 0)),
            "skip_counts": {k: int(v) for k, v in c.items() if k.startswith("skip_")},
        }

    per_ds_metrics = {}
    for dset, c in sorted(per_ds.items()):
        scored = int(c.get("scored", 0))
        per_ds_metrics[dset] = {
            "num_seen": int(c.get("num_seen", 0)),
            "num_scored": scored,
            "accuracy": safe_div(int(c.get("correct", 0)), scored),
        }

    metrics = {
        "jsonl": jsonl_path,
        "data_dir": args.data_dir,
        "split_filter": sf if sf else None,
        "question_category_filter": qf if qf else None,
        "num_items_loaded": len(items),
        "num_items_loaded_raw": num_loaded_raw,
        "num_scored_items": int(overall.get("scored", 0)),
        "accuracy": safe_div(int(overall.get("correct", 0)), int(overall.get("scored", 0))),
        "parse_rate": safe_div(int(overall.get("parse_ok", 0)), int(overall.get("scored", 0))),
        "pred_in_options_rate": safe_div(int(overall.get("pred_in_options", 0)), int(overall.get("scored", 0))),
        "skip_reason_counts": {k: int(v) for k, v in skip.items()},
        "per_question_category": per_q_metrics,
        "per_dataset": per_ds_metrics,
        "predictions_jsonl": pred_jsonl_path,
        "state_load": {
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nSaved metrics to:", result_path)
    print("Saved per-sample predictions to:", pred_jsonl_path)
    print("Done.")


if __name__ == "__main__":
    main()
