import argparse
import json
import os
import re
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


def _per_class_stats(pairs: List[Tuple[str, Optional[str]]]) -> Dict[str, Dict[str, float]]:
    """Return per-class TP/FP/FN and derived precision/recall/F1."""
    labels = sorted({gt for gt, _ in pairs})
    result = {}
    for lbl in labels:
        tp = sum(1 for gt, pr in pairs if gt == lbl and pr == lbl)
        fp = sum(1 for gt, pr in pairs if gt != lbl and pr == lbl)
        fn = sum(1 for gt, pr in pairs if gt == lbl and pr != lbl)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[lbl] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return result


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


def weighted_f1(pairs: List[Tuple[str, Optional[str]]]) -> Optional[float]:
    """F1 weighted by class support (# true instances per class)."""
    if not pairs:
        return None
    stats = _per_class_stats(pairs)
    total = sum(s["support"] for s in stats.values())
    if not total:
        return None
    return sum(s["f1"] * s["support"] for s in stats.values()) / total


def balanced_accuracy(pairs: List[Tuple[str, Optional[str]]]) -> Optional[float]:
    """Macro-averaged recall (= balanced accuracy)."""
    if not pairs:
        return None
    stats = _per_class_stats(pairs)
    if not stats:
        return None
    return sum(s["recall"] for s in stats.values()) / len(stats)


def mcc(pairs: List[Tuple[str, Optional[str]]]) -> Optional[float]:
    """Matthews Correlation Coefficient — meaningful for binary and multi-class."""
    if not pairs:
        return None
    labels = sorted({gt for gt, _ in pairs})
    n = len(pairs)
    # Build confusion matrix
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for gt, pr in pairs:
        if pr is None or pr not in label_idx:
            continue
        cm[label_idx[gt]][label_idx[pr]] += 1
    # MCC via the covariance formula
    t_k = [sum(cm[i]) for i in range(k)]          # true class totals
    p_k = [sum(cm[i][j] for i in range(k)) for j in range(k)]  # predicted class totals
    c   = sum(cm[i][i] for i in range(k))          # correct predictions
    s   = sum(t_k)                                  # total samples
    cov_yy = sum(t_k[i] * (s - t_k[i]) for i in range(k))
    cov_pp = sum(p_k[i] * (s - p_k[i]) for i in range(k))
    if cov_yy == 0 or cov_pp == 0:
        return None
    num = c * s - sum(t_k[i] * p_k[i] for i in range(k))
    return num / ((cov_yy * cov_pp) ** 0.5)


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
    per_dataset  = defaultdict(Counter)
    overall_pairs: List[Tuple[str, Optional[str]]] = []
    per_question_pairs: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    per_dataset_pairs:  Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    overall_gen: List[Tuple[str, str]] = []
    per_question_gen: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    per_dataset_gen:  Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for row in load_jsonl(args.predictions):
        meta = row.get("meta", {}) or {}
        question_category = meta.get("question_category", "UNKNOWN")
        dataset_name      = meta.get("dataset", "UNKNOWN")
        parse_info = row.get("parse", {}) or {}
        gt_ans = (row.get("gt", {}) or {}).get("answer")
        pred_ans = (row.get("pred", {}) or {}).get("answer")
        raw_text = (row.get("raw_text") or "").strip()

        overall["num_predictions"] += 1
        per_question[question_category]["num_predictions"] += 1
        per_dataset[dataset_name]["num_predictions"] += 1

        if row.get("is_correct", False):
            overall["correct"] += 1
            per_question[question_category]["correct"] += 1
            per_dataset[dataset_name]["correct"] += 1

        if parse_info.get("raw_extracted") is not None:
            overall["parse_ok"] += 1
            per_question[question_category]["parse_ok"] += 1
            per_dataset[dataset_name]["parse_ok"] += 1

        if parse_info.get("in_options", False):
            overall["pred_in_options"] += 1
            per_question[question_category]["pred_in_options"] += 1
            per_dataset[dataset_name]["pred_in_options"] += 1

        if parse_info.get("ok_answer_tag", False):
            overall["strict_answer_tag"] += 1
            per_question[question_category]["strict_answer_tag"] += 1
            per_dataset[dataset_name]["strict_answer_tag"] += 1

        if gt_ans:
            overall_pairs.append((gt_ans, pred_ans))
            per_question_pairs[question_category].append((gt_ans, pred_ans))
            per_dataset_pairs[dataset_name].append((gt_ans, pred_ans))

        # Strip <answer> tags so ROUGE-L/BLEU-1 compare clean text vs clean text
        clean_pred = re.sub(r"</?answer>", "", raw_text, flags=re.IGNORECASE).strip()
        if clean_pred and gt_ans:
            overall_gen.append((clean_pred, gt_ans))
            per_question_gen[question_category].append((clean_pred, gt_ans))
            per_dataset_gen[dataset_name].append((clean_pred, gt_ans))

    metrics = {
        "predictions_jsonl": args.predictions,
        "num_predictions": int(overall["num_predictions"]),
        "accuracy": safe_div(overall["correct"], overall["num_predictions"]),
        "balanced_accuracy": balanced_accuracy(overall_pairs),
        "macro_f1": macro_f1(overall_pairs),
        "weighted_f1": weighted_f1(overall_pairs),
        "mcc": mcc(overall_pairs),
        "rouge_l": corpus_rouge_l(overall_gen),
        "bleu1": corpus_bleu1(overall_gen),
        "parse_rate": safe_div(overall["parse_ok"], overall["num_predictions"]),
        "pred_in_options_rate": safe_div(overall["pred_in_options"], overall["num_predictions"]),
        "strict_answer_tag_rate": safe_div(overall["strict_answer_tag"], overall["num_predictions"]),
        "per_question_category": {},
        "per_dataset": {},
    }

    for question_category, counter in sorted(per_question.items()):
        num_predictions = int(counter["num_predictions"])
        pairs_q = per_question_pairs[question_category]
        metrics["per_question_category"][question_category] = {
            "num_predictions": num_predictions,
            "accuracy": safe_div(counter["correct"], num_predictions),
            "balanced_accuracy": balanced_accuracy(pairs_q),
            "macro_f1": macro_f1(pairs_q),
            "weighted_f1": weighted_f1(pairs_q),
            "mcc": mcc(pairs_q),
            "rouge_l": corpus_rouge_l(per_question_gen[question_category]),
            "bleu1": corpus_bleu1(per_question_gen[question_category]),
            "parse_rate": safe_div(counter["parse_ok"], num_predictions),
            "pred_in_options_rate": safe_div(counter["pred_in_options"], num_predictions),
            "strict_answer_tag_rate": safe_div(counter["strict_answer_tag"], num_predictions),
            "per_class": _per_class_stats(pairs_q),
        }

    for dataset_name, counter in sorted(per_dataset.items()):
        num_predictions = int(counter["num_predictions"])
        pairs_d = per_dataset_pairs[dataset_name]
        metrics["per_dataset"][dataset_name] = {
            "num_predictions": num_predictions,
            "accuracy": safe_div(counter["correct"], num_predictions),
            "balanced_accuracy": balanced_accuracy(pairs_d),
            "macro_f1": macro_f1(pairs_d),
            "weighted_f1": weighted_f1(pairs_d),
            "mcc": mcc(pairs_d),
            "rouge_l": corpus_rouge_l(per_dataset_gen[dataset_name]),
            "bleu1": corpus_bleu1(per_dataset_gen[dataset_name]),
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
