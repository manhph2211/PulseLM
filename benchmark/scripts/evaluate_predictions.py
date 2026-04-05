
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_div(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    overall = Counter()
    per_question = defaultdict(Counter)

    for row in load_jsonl(args.predictions):
        meta = row.get("meta", {}) or {}
        question_category = meta.get("question_category", "UNKNOWN")
        parse_info = row.get("parse", {}) or {}

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

    metrics = {
        "predictions_jsonl": args.predictions,
        "num_predictions": int(overall["num_predictions"]),
        "accuracy": safe_div(overall["correct"], overall["num_predictions"]),
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
