"""
aksaraLLM — Mesin Penerjemah Massal v2 (Turbo Batch Edition)

Fitur:
  - Batch translation (10x lebih cepat dari v1!)
  - GPU-accelerated dengan Helsinki-NLP
  - Auto-resume kalau Colab mati
  - Auto-save ke Google Drive
  - Progress bar yang jelas (nggak stuck lagi!)

Usage di Colab:
  # Shard 1 (Colab kedua)
  !python translate_v2.py --shard 1 --total-shards 2 --drive

  # Shard 2 (Colab ketiga)
  !python translate_v2.py --shard 2 --total-shards 2 --drive
"""
import os
import json
import time
import argparse
import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


def translate_batch(texts: list[str], model, tokenizer, device, max_len=512) -> list[str]:
    """Terjemahkan BANYAK teks sekaligus (batch) — 10x lebih cepat!"""
    if not texts:
        return []
    
    # Potong teks yang kepanjangan
    trimmed = [t[:2000] if len(t) > 2000 else t for t in texts]
    
    try:
        inputs = tokenizer(
            trimmed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_len)
        
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return results
    except Exception as e:
        print(f"\n  ⚠️ Batch error (fallback ke satu-satu): {str(e)[:60]}")
        # Fallback: terjemahkan satu-satu
        results = []
        for t in trimmed:
            try:
                inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                with torch.no_grad():
                    out = model.generate(**inp)
                results.append(tokenizer.decode(out[0], skip_special_tokens=True))
            except:
                results.append("")
        return results


def main():
    parser = argparse.ArgumentParser(description="AksaraLLM Penerjemah Massal v2")
    parser.add_argument("--shard", type=int, required=True, help="Nomor shard (1 atau 2)")
    parser.add_argument("--total-shards", type=int, default=2, help="Total shard")
    parser.add_argument("--batch-size", type=int, default=8, help="Jumlah teks per batch")
    parser.add_argument("--drive", action="store_true", help="Simpan ke Google Drive")
    parser.add_argument("--save-every", type=int, default=500, help="Auto-save tiap N baris")
    args = parser.parse_args()

    # Setup output path
    if args.drive:
        out_dir = "/content/drive/MyDrive/aksaraLLM-data"
    else:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"translated_hh_rlhf_shard_{args.shard}.jsonl")

    print("=" * 60)
    print(f"🤖 Mesin Penerjemah AksaraLLM v2 (Shard {args.shard}/{args.total_shards})")
    print(f"   Batch size: {args.batch_size} | Save tiap: {args.save_every}")
    print(f"   Output: {out_file}")
    print("=" * 60)

    # 1. Download dataset
    print("\n📦 Mengunduh dataset Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    total = len(ds)
    print(f"   Total data seluruh dunia: {total}")

    # 2. Hitung porsi shard
    chunk = total // args.total_shards
    start = (args.shard - 1) * chunk
    end = start + chunk if args.shard < args.total_shards else total
    my_data = ds.select(range(start, end))
    my_total = len(my_data)
    print(f"   ✂️ Shard {args.shard}: index {start} → {end} ({my_total} baris)")

    # 3. Cek progress sebelumnya (resume)
    done = 0
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            done = sum(1 for _ in f)
        if done > 0:
            print(f"   🔄 Melanjutkan dari baris ke-{done + 1}!")

    remaining = my_total - done
    if remaining <= 0:
        print("\n🎉 SHARD INI SUDAH SELESAI 100%!")
        return

    print(f"   📝 Sisa: {remaining} baris perlu diterjemahkan")

    # 4. Load model AI penerjemah ke GPU
    print("\n🧠 Memuat Helsinki-NLP ke GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Helsinki-NLP/opus-mt-en-id"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name()
        print(f"   🔥 GPU: {gpu}")
    else:
        print("   💻 CPU mode (lebih lambat)")

    # 5. MULAI MENERJEMAHKAN (BATCH MODE!)
    print("\n" + "=" * 60)
    print("🚀 TERJEMAHAN DIMULAI!\n")

    start_time = time.time()
    batch_chosen = []
    batch_rejected = []
    batch_indices = []

    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(done, my_total):
            row = my_data[i]
            batch_chosen.append(row["chosen"])
            batch_rejected.append(row["rejected"])
            batch_indices.append(start + i)

            # Proses batch saat penuh
            if len(batch_chosen) >= args.batch_size or i == my_total - 1:
                # Terjemahkan batch sekaligus
                id_chosen = translate_batch(batch_chosen, model, tokenizer, device)
                id_rejected = translate_batch(batch_rejected, model, tokenizer, device)

                # Tulis semua hasil ke file
                for j in range(len(batch_chosen)):
                    line = json.dumps({
                        "chosen": id_chosen[j],
                        "rejected": id_rejected[j],
                        "original_idx": batch_indices[j],
                    }, ensure_ascii=False)
                    f.write(line + "\n")

                # Reset batch
                batch_chosen.clear()
                batch_rejected.clear()
                batch_indices.clear()

                # Progress report
                completed = i + 1
                elapsed = time.time() - start_time
                speed = (completed - done) / elapsed if elapsed > 0 else 0
                eta = (my_total - completed) / speed if speed > 0 else 0

                print(
                    f"  ⚡ {completed}/{my_total} "
                    f"({completed * 100 // my_total}%) | "
                    f"Speed: {speed:.1f} rows/s | "
                    f"ETA: {eta / 60:.1f} min",
                    flush=True,
                )

                # Auto-save (flush ke disk)
                if completed % args.save_every == 0:
                    f.flush()
                    os.fsync(f.fileno())
                    print(f"  💾 Auto-saved @ baris {completed}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 SHARD {args.shard} SELESAI!")
    print(f"   Waktu: {total_time / 60:.1f} menit")
    print(f"   File: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
