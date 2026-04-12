"""
AksaraLLM — Translasi Dataset Paralel dengan Range Baris
=========================================================
Script ini untuk menerjemahkan hh-rlhf secara PARALEL
menggunakan beberapa akun Colab sekaligus.

Contoh pakai 3 akun Colab paralel:
  Akun 1: python translate_range.py --start 35000 --end 53400 --output shard_35k_53k.jsonl --drive
  Akun 2: python translate_range.py --start 53400 --end 66800 --output shard_53k_66k.jsonl --drive
  Akun 3: python translate_range.py --start 66800 --end 80400 --output shard_66k_80k.jsonl --drive

Setelah selesai, merge dengan:
  python merge_shards.py --input "shard_35k*,shard_53k*,shard_66k*" --output translated_hh_rlhf_shard_1_complete.jsonl
"""

import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

MAX_CHARS = 2000


def translate_safe(text: str, model, tokenizer, device) -> str:
    if not text.strip():
        return text
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "..."
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="AksaraLLM — Translasi Range Baris Tertentu (untuk paralel multi-akun)"
    )
    parser.add_argument("--start", type=int, required=True,
                        help="Baris pertama yang akan diterjemahkan (index dataset)")
    parser.add_argument("--end", type=int, required=True,
                        help="Baris terakhir (eksklusif) yang akan diterjemahkan")
    parser.add_argument("--output", type=str, required=True,
                        help="Nama file output JSONL (contoh: shard_50k_65k.jsonl)")
    parser.add_argument("--drive", action="store_true",
                        help="Simpan ke Google Drive (/content/drive/MyDrive/aksaraLLM-data/)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200,
                        help="Flush/save ke disk setiap N baris (default: 200)")
    args = parser.parse_args()

    # Setup output path
    if args.drive:
        out_dir = "/content/drive/MyDrive/aksaraLLM-data"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.output)
    else:
        out_path = args.output

    print("=" * 55)
    print(f"🚀 AksaraLLM Translasi Paralel")
    print(f"   Range  : baris {args.start} → {args.end}")
    print(f"   Output : {out_path}")
    print("=" * 55)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print("\n📦 Mengunduh dataset Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"   Total dataset: {len(ds):,} baris")

    # Ambil porsi yang diminta
    end_idx = min(args.end, len(ds))
    portion = ds.select(range(args.start, end_idx))
    total_to_process = len(portion)
    print(f"   Porsi ini: {args.start:,} → {end_idx:,} ({total_to_process:,} baris)")

    # Auto-resume: hitung sudah berapa baris yang diterjemahkan
    already_done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            already_done = sum(1 for _ in f)
        print(f"\n🔄 Auto-resume: sudah {already_done:,} baris, lanjut dari sini!")
    remaining = total_to_process - already_done
    print(f"   Sisa: {remaining:,} baris")

    if remaining <= 0:
        print("\n✅ Semua baris sudah diterjemahkan!")
        return

    # Load model Helsinki-NLP
    print("\n🧠 Memuat Helsinki-NLP ke GPU...")
    model_name = "Helsinki-NLP/opus-mt-en-id"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()
    print("   ✅ Model siap!")

    # Mulai translasi
    print(f"\n{'='*55}")
    print("🚀 TERJEMAHAN DIMULAI!")
    print(f"{'='*55}\n")

    import time
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(already_done, total_to_process):
            row = portion[i]

            id_chosen   = translate_safe(row["chosen"],   model, tokenizer, device)
            id_rejected = translate_safe(row["rejected"], model, tokenizer, device)

            json_line = json.dumps({
                "chosen":       id_chosen,
                "rejected":     id_rejected,
                "original_idx": args.start + i,
            }, ensure_ascii=False)
            f.write(json_line + "\n")

            # Progress log
            if (i + 1) % 16 == 0:
                done      = i + 1 - already_done
                elapsed   = time.time() - t0
                speed     = done / elapsed if elapsed > 0 else 0
                eta_s     = (remaining - done) / speed if speed > 0 else 0
                print(
                    f"  ★ {args.start + i + 1}/{end_idx} "
                    f"| Speed: {speed:.1f} rows/s "
                    f"| ETA: {eta_s/60:.1f} min",
                    flush=True
                )

            # Flush ke disk secara berkala
            if (i + 1) % args.save_every == 0:
                f.flush()
                os.fsync(f.fileno())
                print(f"  💾 Auto-saved ({i + 1 - already_done} baris baru tersimpan)")

    print(f"\n{'='*55}")
    print(f"🎉 SELESAI! Baris {args.start} → {end_idx}")
    print(f"   File: {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
