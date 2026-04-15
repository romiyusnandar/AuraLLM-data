import argparse
import json
import re
from pathlib import Path


HUMAN_PATTERNS = [
    re.compile(r"\bHuman\s*:", re.IGNORECASE),
    re.compile(r"\bUser\s*:", re.IGNORECASE),
]

ASSISTANT_PATTERNS = [
    re.compile(r"\bAssistant\s*:", re.IGNORECASE),
    re.compile(r"\bAsst\s*:", re.IGNORECASE),
]


def normalize_role_labels(text: str) -> str:
    if not isinstance(text, str):
        return text

    result = text
    for pattern in HUMAN_PATTERNS:
        result = pattern.sub("Manusia:", result)
    for pattern in ASSISTANT_PATTERNS:
        result = pattern.sub("Asisten:", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Normalisasi label Human/Assistant pada file JSONL."
    )
    parser.add_argument("--input", required=True, help="File JSONL input")
    parser.add_argument("--output", required=True, help="File JSONL output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[skip] JSON tidak valid di line {line_no}")
                continue

            total += 1
            before_chosen = record.get("chosen", "")
            before_rejected = record.get("rejected", "")

            record["chosen"] = normalize_role_labels(before_chosen)
            record["rejected"] = normalize_role_labels(before_rejected)

            if record["chosen"] != before_chosen or record["rejected"] != before_rejected:
                changed += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Selesai. Total record: {total}")
    print(f"Label diperbarui: {changed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
