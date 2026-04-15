import argparse
import glob
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[skip] JSON tidak valid: {path} line {line_no}")


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def should_keep(record, drop_equal, drop_empty):
    chosen = normalize_text(record.get("chosen", ""))
    rejected = normalize_text(record.get("rejected", ""))

    if drop_empty and (not chosen or not rejected):
        return False, "empty"

    if drop_equal and chosen == rejected:
        return False, "equal"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Gabungkan file JSONL hasil translasi dan filter record yang jelek."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Glob input, contoh: "part_*.jsonl"',
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Nama file output JSONL final",
    )
    parser.add_argument(
        "--drop-equal",
        action="store_true",
        help="Buang record jika chosen == rejected setelah normalisasi spasi",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Buang record jika chosen atau rejected kosong",
    )
    parser.add_argument(
        "--dedup-by-idx",
        action="store_true",
        help="Buang duplikat berdasarkan original_idx",
    )
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.input))
    if not paths:
        raise SystemExit(f"Tidak ada file yang cocok dengan pattern: {args.input}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_idx = set()
    stats = {
        "read": 0,
        "written": 0,
        "skip_equal": 0,
        "skip_empty": 0,
        "skip_dup_idx": 0,
    }

    with output_path.open("w", encoding="utf-8") as f_out:
        for path in paths:
            print(f"[read] {path}")
            for record in load_jsonl(path):
                stats["read"] += 1

                keep, reason = should_keep(record, args.drop_equal, args.drop_empty)
                if not keep:
                    if reason == "equal":
                        stats["skip_equal"] += 1
                    elif reason == "empty":
                        stats["skip_empty"] += 1
                    continue

                if args.dedup_by_idx:
                    original_idx = record.get("original_idx")
                    if original_idx in seen_idx:
                        stats["skip_dup_idx"] += 1
                        continue
                    seen_idx.add(original_idx)

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1

    print("\nSelesai.")
    print(f"Read       : {stats['read']}")
    print(f"Written    : {stats['written']}")
    print(f"Skip equal : {stats['skip_equal']}")
    print(f"Skip empty : {stats['skip_empty']}")
    print(f"Skip dup   : {stats['skip_dup_idx']}")
    print(f"Output     : {output_path}")


if __name__ == "__main__":
    main()
