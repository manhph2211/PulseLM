import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_div(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def macro_f1(pairs: List[Tuple[str, Optional[str]]]) -> Optional[float]:
    labels = sorted({gt for gt, _ in pairs})
    if not labels:
        return None
    f1s = []
    for lbl in labels:
        tp = sum(1 for gt, pr in pairs if gt == lbl and pr == lbl)
        fp = sum(1 for gt, pr in pairs if gt != lbl and pr == lbl)
        fn = sum(1 for gt, pr in pairs if gt == lbl and pr != lbl)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    matches = sum(min(cnt, ref_counts[tok]) for tok, cnt in Counter(pred_tokens).items())
    return matches / len(pred_tokens)


def corpus_rouge_l(pairs: List[Tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    return sum(rouge_l(p, r) for p, r in pairs) / len(pairs)


def corpus_bleu1(pairs: List[Tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    return sum(bleu1(p, r) for p, r in pairs) / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    overall = Counter()
    per_question = defaultdict(Counter)
    overall_pairs: List[Tuple[str, Optional[str]]] = []
    per_question_pairs: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    overall_gen: List[Tuple[str, str]] = []
    per_question_gen: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for row in load_jsonl(args.predictions):
        meta = row.get("meta", {}) or {}
        question_category = meta.get("question_category", "UNKNOWN")
        parse_info = row.get("parse", {}) or {}
        gt_ans = (row.get("gt", {}) or {}).get("answer")
        pred_ans = (row.get("pred", {}) or {}).get("answer")
        raw_text = (row.get("raw_text") or "").strip()

        overall["num_predictions"] += 1
        per_question[question_category]["num_predictions"] += 1

        if row.get("is_correct", False):
            overall["correct"] += 1
            per_question[question_category]["correct"] += 1

        if parse_info.get("raw_extracted") is not None:
            overall["parse_ok"] += 1
            per_question[question_category]["parse_ok"] += 1

        if parse_info.get("in_options", False):
            overall["pred_in_options"] += 1
            per_question[question_category]["pred_in_options"] += 1

        if parse_info.get("ok_answer_tag", False):
            overall["strict_answer_tag"] += 1
            per_question[question_category]["strict_answer_tag"] += 1

        if gt_ans:
            overall_pairs.append((gt_ans, pred_ans))
            per_question_pairs[question_category].append((gt_ans, pred_ans))

        if raw_text and gt_ans:
            overall_gen.append((raw_text, gt_ans))
            per_question_gen[question_category].append((raw_text, gt_ans))

    metrics = {
        "predictions_jsonl": args.predictions,
        "num_predictions": int(overall["num_predictions"]),
        "accuracy": safe_div(overall["correct"], overall["num_predictions"]),
        "macro_f1": macro_f1(overall_pairs),
        "rouge_l": corpus_rouge_l(overall_gen),
        "bleu1": corpus_bleu1(overall_gen),
        "parse_rate": safe_div(overall["parse_ok"], overall["num_predictions"]),
        "pred_in_options_rate": safe_div(overall["pred_in_options"], overall["num_predictions"]),
        "strict_answer_tag_rate": safe_div(overall["strict_answer_tag"], overall["num_predictions"]),
        "per_question_category": {},
    }

    for question_category, counter in sorted(per_question.items()):
        num_predictions = int(counter["num_predictions"])
        metrics["per_question_category"][question_category] = {
            "num_predictions": num_predictions,
            "accuracy": safe_div(counter["correct"], num_predictions),
            "macro_f1": macro_f1(per_question_pairs[question_category]),
            "rouge_l": corpus_rouge_l(per_question_gen[question_category]),
            "bleu1": corpus_bleu1(per_question_gen[question_category]),
            "parse_rate": safe_div(counter["parse_ok"], num_predictions),
            "pred_in_options_rate": safe_div(counter["pred_in_options"], num_predictions),
            "strict_answer_tag_rate": safe_div(counter["strict_answer_tag"], num_predictions),
        }

    out_path = os.path.join(args.out_dir, "prediction_metrics.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
