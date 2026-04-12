import argparse
import sys
import os
import json
import time
import subprocess
from datasets import load_dataset
from types import SimpleNamespace
import torch
from transformers import MarianMTModel, MarianTokenizer

# Konstanta batas panjang karakter (Helsinki model seq len limit)
MAX_CHARS = 2000 

def translate_safe(text: str, model, tokenizer, device) -> str:
    """Menerjemahkan teks dengan aman menggunakan model AI offline lokal."""
    if not text.strip():
        return text
        
    # Kalau kepanjangan, potong saja demi keamanan RAM Tensor
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "..."
        
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"\n  [!] Gagal menerjemahkan baris ini (Skip): {str(e)[:50]}...")
        return ""

def push_to_github(file_path: str, shard: int, start_idx: int, end_idx: int, token: str):
    """Fungsi ajaib untuk nyimpen kerjaan dari Colab langsung ke Github aksara-data."""
    print("\n🚀 Menyimpan otomatis ke GitHub (Auto-Save)...")
    
    # Setup kredensial Github sekali tembak
    os.system(f'git remote set-url origin https://{token}@github.com/aksaraLLM/aksara-data.git')
    os.system('git config --global user.email "bot-aksara@example.com"')
    os.system('git config --global user.name "Aksara-Bot-Penerjemah"')
    
    # Eksekusi Git Commit & Push
    os.system(f'git add {file_path}')
    os.system(f'git commit -m "data(translate): Upload hasil shard {shard} baris {start_idx}-{end_idx}"')
    
    res = os.system('git push origin main')
    if res == 0:
        print("✅ Berhasil di-Push ke GitHub!\n")
    else:
        print("❌ Gagal Push ke Github (Mungkin token salah atau koneksi jelek).\n")

def main():
    parser = argparse.ArgumentParser(description="AksaraLLM - Penerjemah Dataset Massal Terdistribusi")
    parser.add_argument("--shard", type=int, required=True, help="Akun Colab ke-berapa ini? (Contoh: 1, 2, atau 3)")
    parser.add_argument("--total-shards", type=int, required=True, help="Total barisan pekerja Colab (Misal: 3)")
    parser.add_argument("--github-token", type=str, default="", help="Personal Access Token Github milikmu buat Auto-Save")
    parser.add_argument("--save-every", type=int, default=1000, help="Simpan ke github tiap berapa baris?")
    args = parser.parse_args()

    print(f"🤖 Menyalakan Mesin Penerjemah AksaraLLM (Pekerja ke-{args.shard} dari {args.total_shards})")
    print("-" * 50)
    
    # 1. Download Dataset Asli (Anthropic punya 160k baris)
    print("📦 Mengunduh Dataset Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    total_data = len(ds)
    print(f"Total baris seluruh dunia: {total_data}")

    # 2. Proses Sharding (Membagi porsi kerjaan agar adil untuk ke-3 Colab)
    # Misal total_data 160.000 / 3 = ~53.333 baris per Colab
    chunk_size = total_data // args.total_shards
    start_idx = (args.shard - 1) * chunk_size
    # Shard terakhir mengambil semua sisanya
    end_idx = start_idx + chunk_size if args.shard < args.total_shards else total_data
    
    print(f"✂️ Akun Colab ini (Shard {args.shard}) kebagian Menerjemahkan Indeks: {start_idx} sampai {end_idx}")
    
    # 3. Ambil porsinya saja
    my_portion = ds.select(range(start_idx, end_idx))
    
    # Siapkan output JSONL
    out_file = f"translated_hh_rlhf_shard_{args.shard}.jsonl"
    print(f"💾 Hasil terjemahan akan dicicil ke file: {out_file}")
    
    # Cari tahu sudah sampai mana kalau misalnya kemarin script mati di tengah jalan
    processed_count = 0
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
        print(f"🔄 Melanjutkan dari baris ke-{processed_count+1} (Menghemat waktu!)")
    
    # Setup offline translation RAW API (Bypass pipeline bug)
    print("🧠 Memuat Model AI Penerjemah Lokal (Helsinki-NLP) ke GPU...")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    model_name = "Helsinki-NLP/opus-mt-en-id"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    
    print("\n" + "-" * 50)
    print("🚀 PROSES TRANSLATE DIMULAI! (Berjalan sunyi di GPU)\n")
    
    # 4. MULAI MENERJEMAHKAN!
    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(processed_count, len(my_portion)):
            row = my_portion[i]
            
            # Anthropic punya 2 kolom utama: "chosen" (bagus) & "rejected" (jelek)
            en_chosen = row['chosen']
            en_rejected = row['rejected']
            
            # Terjemahkan keduanya
            if i % 10 == 0:
                print(f"⚡ [Shard {args.shard}] Selesai menerjemahkan s/d baris {start_idx + i} / {end_idx}...", flush=True)
                
            id_chosen = translate_safe(en_chosen, model, tokenizer, device)
            id_rejected = translate_safe(en_rejected, model, tokenizer, device)
            
            # Simpan dengan format JSONL (1 baris JSON per percakapan)
            json_line = json.dumps({
                "chosen": id_chosen,
                "rejected": id_rejected,
                "original_idx": start_idx + i
            }, ensure_ascii=False)
            
            f.write(json_line + "\n")
            
            # AUTO-SAVE KE GITHUB (Biar aman!)
            if (i + 1) % args.save_every == 0:
                print(f"\n📂 Auto-Save tercapai di baris {(i+1)}. Menyimpan ke file lokal...")
                f.flush()
                os.fsync(f.fileno()) # Memastikan tersimpan murni ke hardisk
                
                if args.github_token:
                    push_to_github(out_file, args.shard, start_idx + i - args.save_every, start_idx + i, args.github_token)
    
    # Selesai semuanya! Push terakhir
    print(f"\n🎉 SHARD {args.shard} SELESAI SEMUA! 100% SUCCESS!")
    if args.github_token:
        push_to_github(out_file, args.shard, start_idx, end_idx, args.github_token)

if __name__ == "__main__":
    main()
