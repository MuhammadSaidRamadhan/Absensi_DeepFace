import os
import sys
import numpy as np
import joblib
import json
import csv
import sqlite3
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS

# --- Konfigurasi ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.utils import extract_face_features

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
MODEL_DIR = PROJECT_ROOT / "backend" / "model"
AUDIO_FILES_DIR = PROJECT_ROOT / "backend" / "generated_audio"
DB_PATH = PROJECT_ROOT / "backend" / "attendance.db"
INTERNS_CSV_PATH = PROJECT_ROOT / "interns.csv"
AUDIO_TRACKING_FILE = PROJECT_ROOT / "backend" / "audio_tracking.json"

# --- FUNGSI DATABASE (Tidak ada perubahan) ---
def create_or_update_local_db():
    print("\n💾 Memeriksa dan menyinkronkan database lokal...")
    if not INTERNS_CSV_PATH.exists():
        print(f"❌ Error: File 'interns.csv' tidak ditemukan."); return False
    try:
        with open(INTERNS_CSV_PATH, mode='r', encoding='utf-8') as f:
            local_data = {row['name']: row for row in csv.DictReader(f)}
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS interns (id INTEGER PRIMARY KEY, name TEXT UNIQUE, universitas TEXT, kategori TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS attendance_logs (id INTEGER PRIMARY KEY, intern_id INTEGER, intern_name TEXT, universitas TEXT, kategori TEXT, image_url TEXT, absent_at TEXT)")
        cursor.execute("SELECT name FROM interns")
        db_names = {row[0] for row in cursor.fetchall()}
        new_names = set(local_data.keys()) - db_names
        if new_names:
            print(f"   ✨ Menambahkan {len(new_names)} data baru: {new_names}")
            for name in new_names:
                data = local_data[name]
                cursor.execute("INSERT INTO interns (name, universitas, kategori) VALUES (?, ?, ?)", (data['name'], data['universitas'], data['kategori']))
        conn.commit(); conn.close()
        print("   ✅ Struktur database SIAP."); return True
    except Exception as e:
        print(f"   ❌ Gagal menyinkronkan database: {e}"); return False

# --- FUNGSI AUDIO (REVISI BESAR) ---
def generate_all_audio_files(labels):
    print("\n🎵 Memeriksa dan membuat semua file audio...")
    AUDIO_FILES_DIR.mkdir(exist_ok=True)
    
    # 1. Buat Audio Umum (untuk status sistem)
    generic_messages = {
        "S001": "Anda Sudah Melakukan Absensi Hari Ini",
        "S002": "Wajah Tidak Terdeteksi, Silahkan Coba Lagi",
        "S003": "Data Wajah Anda Tidak Ditemukan Di Sistem"
    }
    print("   🔊 Membuat audio umum...")
    for track_id, message in generic_messages.items():
        audio_path = AUDIO_FILES_DIR / f"{track_id}.mp3"
        if not audio_path.exists():
            try:
                gTTS(text=message, lang='id').save(str(audio_path))
                print(f"      ✅ Berhasil membuat {track_id}.mp3")
            except Exception as e:
                print(f"      ❌ Gagal membuat {track_id}.mp3: {e}")

    # 2. Buat Audio per Nama (untuk sambutan)
    print("\n   👤 Membuat audio per nama...")
    try:
        if AUDIO_TRACKING_FILE.exists():
            with open(AUDIO_TRACKING_FILE, 'r') as f: audio_tracking = json.load(f)
        else:
            audio_tracking = {}
        
        track_number = max([int(v) for v in audio_tracking.values() if v.isdigit()] + [0]) + 1
        new_audio_generated = False

        for label in sorted(list(labels)):
            if label not in audio_tracking:
                audio_path = AUDIO_FILES_DIR / f"{track_number:04d}.mp3"
                gTTS(text=f"Absensi Berhasil, Selamat datang {label}", lang='id').save(str(audio_path))
                print(f"      ✅ Berhasil membuat audio untuk {label} ({track_number:04d}.mp3)")
                audio_tracking[label] = f"{track_number:04d}"
                track_number += 1
                new_audio_generated = True

        if new_audio_generated:
            with open(AUDIO_TRACKING_FILE, 'w') as f: json.dump(audio_tracking, f, indent=2)
            print("   💾 Audio tracking untuk nama berhasil disimpan.")
        else:
            print("   ✅ Semua file audio nama sudah lengkap.")
    except Exception as e:
        print(f"   ❌ Gagal membuat audio per nama: {e}. Pastikan terkoneksi internet.")

# --- FUNGSI TRAINING (Tidak ada perubahan) ---
def train_model_full():
    print("\n🧠 Memulai proses training penuh dengan DeepFace...")
    embeddings, labels = [], []
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir(): continue
        person_name = person_dir.name
        img_count = 0
        for img_path in person_dir.glob("*.jpg"):
            emb_list = extract_face_features(str(img_path))
            if emb_list:
                embeddings.extend(emb_list)
                labels.extend([person_name] * len(emb_list))
                img_count += len(emb_list)
        print(f"   👤 {person_name}: {img_count} wajah diekstrak.")
    if not embeddings:
        print("\n❌ Tidak ada wajah yang berhasil diekstrak!"); return False, []
    
    embeddings, labels = np.array(embeddings), np.array(labels)
    label_encoder = LabelEncoder().fit(labels)
    print(f"\nDEBUG [Training]: LabelEncoder dilatih dengan kelas -> {list(label_encoder.classes_)}\n")
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(embeddings, label_encoder.transform(labels))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(knn, MODEL_DIR / "knn_model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print(f"\n✅ Training selesai! Model disimpan."); return True, set(labels)

# --- FUNGSI UTAMA (Panggil fungsi audio yang baru) ---
def main():
    print("="*50); print("🤖 SCRIPT TRAINING (ENGINE: DEEPFACE)"); print("="*50)
    if not DATASET_DIR.is_dir():
        print(f"❌ Dataset tidak ditemukan."); return
    try:
        success, unique_labels = train_model_full()
        if success:
            generate_all_audio_files(unique_labels) # <-- Panggil fungsi audio yang baru
            create_or_update_local_db()
            print("\n🎉 Semua proses (Training, Audio, & DB) selesai!")
        else:
            print("\n❌ Proses training gagal.")
    except Exception as e:
        print(f"\n❌ Terjadi error tak terduga: {e}")

if __name__ == "__main__":
    main()