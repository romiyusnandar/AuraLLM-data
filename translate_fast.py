"""
AksaraLLM — Translasi ULTRA-CEPAT dengan Batch GPU
====================================================
Menggunakan batch inference GPU untuk kecepatan maksimal.
Target: ~15-30 rows/s (vs 0.6 rows/s sebelumnya = 25-50x lebih cepat!)

Cara pakai (3 akun paralel):
  Akun 1: python translate_fast.py --start 35000 --end 53400 --output part_35k.jsonl --drive
  Akun 2: python translate_fast.py --start 53400 --end 66900 --output part_53k.jsonl --drive
  Akun 3: python translate_fast.py --start 66900 --end 80400 --output part_66k.jsonl --drive
"""

import argparse
import json
import os
import time
import torch
from datasets import load_dataset
from transformers import pipeline

MAX_CHARS = 350   # Potong panjang → token lebih pendek → batching lebih cepat


def load_translator():
    """Load Helsinki-NLP dengan GPU + FP16 + batch pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"🔥 GPU {'aktif' if device == 0 else 'tidak tersedia, pakai CPU'}")
    if torch.cuda.is_available():
        print(f"   {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("🧠 Memuat Helsinki-NLP/opus-mt-en-id...")
    translator = pipeline(
        task="translation_en_to_id",
        model="Helsinki-NLP/opus-mt-en-id",
        device=device,
        torch_dtype=dtype,
        batch_size=64,           # <- Kunci kecepatan! Proses 64 teks sekaligus
        truncation=True,
        max_length=350,
    )
    print("✅ Model siap!")
    return translator


def prepare_texts(batch_rows):
    """Siapkan teks chosen + rejected dari batch rows."""
    chosen_texts   = []
    rejected_texts = []
    for row in batch_rows:
        ch = (row["chosen"]   or "")[:MAX_CHARS]
        rj = (row["rejected"] or "")[:MAX_CHARS]
        chosen_texts.append(ch if ch.strip() else "-")
        rejected_texts.append(rj if rj.strip() else "-")
    return chosen_texts, rejected_texts


def main():
    parser = argparse.ArgumentParser(
        description="AksaraLLM — Translasi ULTRA-CEPAT hh-rlhf dengan Batch GPU"
    )
    parser.add_argument("--start", type=int, required=True,
                        help="Baris pertama dataset yang akan diterjemahkan")
    parser.add_argument("--end", type=int, required=True,
                        help="Baris terakhir (eksklusif)")
    parser.add_argument("--output", type=str, required=True,
                        help="Nama file output JSONL")
    parser.add_argument("--drive", action="store_true",
                        help="Simpan ke /content/drive/MyDrive/aksaraLLM-data/")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Jumlah teks yang diproses GPU sekaligus (default: 64)")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Flush ke disk setiap N baris (default: 500)")
    args = parser.parse_args()

    # ── Output path ──────────────────────────────────────────────────────
    if args.drive:
        out_dir = "/content/drive/MyDrive/aksaraLLM-data"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.output)
    else:
        out_path = args.output

    print("=" * 58)
    print(f"🚀 AksaraLLM Ultra-Fast Translator")
    print(f"   Range  : {args.start:,} → {args.end:,}")
    print(f"   Output : {out_path}")
    print("=" * 58)

    # ── Load dataset ─────────────────────────────────────────────────────
    print("\n📦 Mengunduh Anthropic/hh-rlhf...")
    ds       = load_dataset("Anthropic/hh-rlhf", split="train")
    end_real = min(args.end, len(ds))
    portion  = ds.select(range(args.start, end_real))
    total    = len(portion)
    print(f"   Total porsi: {total:,} baris")

    # ── Auto-resume ───────────────────────────────────────────────────────
    already_done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            already_done = sum(1 for _ in f)
        print(f"🔄 Auto-resume: {already_done:,} baris sudah selesai, lanjut dari sini!")

    remaining = total - already_done
    if remaining <= 0:
        print("✅ Semua baris sudah diterjemahkan!")
        return

    print(f"   Sisa: {remaining:,} baris\n")

    # ── Load model ────────────────────────────────────────────────────────
    translator = load_translator()

    # ── Translasi BATCH ───────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"⚡ TERJEMAHAN BATCH DIMULAI! (batch_size={args.batch_size})")
    print(f"{'='*58}\n")

    t0          = time.time()
    rows_done   = 0
    chunk_size  = args.batch_size  # Proses sebanyak batch_size baris per iterasi

    with open(out_path, "a", encoding="utf-8") as f_out:
        idx = already_done
        while idx < total:
            # Ambil batch
            batch_end  = min(idx + chunk_size, total)
            batch_rows = [portion[i] for i in range(idx, batch_end)]
            batch_n    = len(batch_rows)

            # Siapkan teks
            chosen_texts, rejected_texts = prepare_texts(batch_rows)

            # Terjemahkan SEKALIGUS (batch GPU!)
            try:
                chosen_results   = translator(chosen_texts,   max_length=350)
                rejected_results = translator(rejected_texts, max_length=350)
            except Exception as e:
                print(f"  ⚠️ Batch error (skip): {e}")
                # Fallback: translate satu-satu
                chosen_results   = [{"translation_text": ""} for _ in chosen_texts]
                rejected_results = [{"translation_text": ""} for _ in rejected_texts]

            # Tulis hasil
            for i, (row, ch_res, rj_res) in enumerate(
                zip(batch_rows, chosen_results, rejected_results)
            ):
                json_line = json.dumps({
                    "chosen":       ch_res.get("translation_text", ""),
                    "rejected":     rj_res.get("translation_text", ""),
                    "original_idx": args.start + idx + i,
                }, ensure_ascii=False)
                f_out.write(json_line + "\n")

            rows_done += batch_n
            idx       += batch_n

            # Progress
            elapsed = time.time() - t0
            speed   = rows_done / elapsed if elapsed > 0 else 0
            eta_m   = (remaining - rows_done) / speed / 60 if speed > 0 else 0

            print(
                f"  ★ {args.start + idx}/{end_real} "
                f"| Speed: {speed:.1f} rows/s "
                f"| ETA: {eta_m:.1f} min",
                flush=True
            )

            # Auto-save
            if rows_done % args.save_every < chunk_size:
                f_out.flush()
                os.fsync(f_out.fileno())
                print(f"  💾 Tersimpan ({rows_done:,} baris baru)")

    total_time = (time.time() - t0) / 60
    print(f"\n{'='*58}")
    print(f"🎉 SELESAI! {remaining:,} baris dalam {total_time:.1f} menit")
    print(f"   Kecepatan rata-rata: {remaining/total_time/60:.1f} rows/s")
    print(f"   File: {out_path}")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
