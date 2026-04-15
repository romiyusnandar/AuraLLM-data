import argparse
import json
import re
from pathlib import Path


HUMAN_LABEL_RE = re.compile(r"\b(Human|Assistant)\s*:", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def norm(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def text_flags(text):
    flags = []
    if not text:
        flags.append("empty")
        return flags
    if HUMAN_LABEL_RE.search(text):
        flags.append("english_role_label")
    if EMAIL_RE.search(text):
        flags.append("email_like")
    if PHONE_RE.search(text):
        flags.append("phone_like")
    if len(text) < 20:
        flags.append("too_short")
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Audit kualitas file JSONL hasil translasi."
    )
    parser.add_argument("--input", required=True, help="Path file JSONL")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Maksimal contoh masalah per kategori",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"File tidak ditemukan: {path}")

    stats = {
        "total": 0,
        "invalid_json": 0,
        "equal_pair": 0,
        "chosen_empty": 0,
        "rejected_empty": 0,
        "english_role_label": 0,
        "email_like": 0,
        "phone_like": 0,
        "too_short": 0,
        "missing_original_idx": 0,
    }
    samples = {key: [] for key in stats if key not in {"total"}}

    for line_no, record in iter_jsonl(path):
        stats["total"] += 1

        if record is None:
            stats["invalid_json"] += 1
            if len(samples["invalid_json"]) < args.sample_limit:
                samples["invalid_json"].append(f"line {line_no}")
            continue

        chosen = norm(record.get("chosen", ""))
        rejected = norm(record.get("rejected", ""))

        if "original_idx" not in record:
            stats["missing_original_idx"] += 1
            if len(samples["missing_original_idx"]) < args.sample_limit:
                samples["missing_original_idx"].append(f"line {line_no}")

        if not chosen:
            stats["chosen_empty"] += 1
            if len(samples["chosen_empty"]) < args.sample_limit:
                samples["chosen_empty"].append(f"line {line_no}")

        if not rejected:
            stats["rejected_empty"] += 1
            if len(samples["rejected_empty"]) < args.sample_limit:
                samples["rejected_empty"].append(f"line {line_no}")

        if chosen and rejected and chosen == rejected:
            stats["equal_pair"] += 1
            if len(samples["equal_pair"]) < args.sample_limit:
                samples["equal_pair"].append(f"line {line_no}")

        combined = f"{chosen}\n{rejected}"
        for flag in text_flags(combined):
            stats[flag] += 1
            if len(samples[flag]) < args.sample_limit:
                samples[flag].append(f"line {line_no}")

    print(f"File  : {path}")
    print(f"Total : {stats['total']}")
    print("")

    ordered_keys = [
        "invalid_json",
        "equal_pair",
        "chosen_empty",
        "rejected_empty",
        "english_role_label",
        "email_like",
        "phone_like",
        "too_short",
        "missing_original_idx",
    ]

    for key in ordered_keys:
        print(f"{key:20} : {stats[key]}")
        if samples[key]:
            print(f"  contoh: {', '.join(samples[key])}")


if __name__ == "__main__":
    main()
