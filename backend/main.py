import os
import time
import json
import datetime
import uvicorn
import sqlite3
from threading import Timer
from pathlib import Path
from gtts import gTTS
import cv2
import numpy as np
import joblib
import pytz
import webbrowser # Impor tunggal
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

# Impor fungsi dari utils.py versi deepface kita
from backend.utils import extract_face_features

# --- Konfigurasi ---
class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    DB_PATH = BASE_DIR / "attendance.db"
    MODEL_DIR = BASE_DIR / "model"
    IMAGE_STORAGE_DIR = BASE_DIR / "captured_images"
    # Konfigurasi Baru untuk Frontend
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    STATIC_DIR = PROJECT_ROOT / "static"
    
    KNN_MODEL_PATH = MODEL_DIR / "knn_model.pkl"
    LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
    AUDIO_TRACKING_FILE = BASE_DIR / "audio_tracking.json"
    TIMEZONE = 'Asia/Jakarta'
    WIB = pytz.timezone(TIMEZONE)

# Pastikan folder ada sebelum server berjalan
AppConfig.IMAGE_STORAGE_DIR.mkdir(exist_ok=True)

# --- Variabel Global & Fungsi Startup ---
knn_model, label_encoder, audio_tracking = None, None, {}
INTERN_CACHE, absen_tercatat = {}, set()

def db_connect():
    """Membuka koneksi baru ke database SQLite."""
    # Pastikan AppConfig.DB_PATH adalah string untuk kompatibilitas sqlite3 yang lebih luas
    conn = sqlite3.connect(str(AppConfig.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def load_all_data():
    global knn_model, label_encoder, audio_tracking, INTERN_CACHE, absen_tercatat
    try:
        # Pengecekan file model harus dilakukan di sini
        if not AppConfig.KNN_MODEL_PATH.exists() or not AppConfig.LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError("Model KNN atau Label Encoder tidak ditemukan.")
            
        knn_model = joblib.load(AppConfig.KNN_MODEL_PATH)
        label_encoder = joblib.load(AppConfig.LABEL_ENCODER_PATH)
        print("✅ Model klasifikasi (KNN) dimuat.")
        
        with open(AppConfig.AUDIO_TRACKING_FILE, 'r') as f: audio_tracking = json.load(f)
        print(f"✅ Pemetaan audio dimuat: {len(audio_tracking)} rekaman.")

        conn = db_connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, universitas, kategori FROM interns")
        for row in cursor.fetchall(): INTERN_CACHE[row['name']] = dict(row)
        print(f"✅ Cache data intern dimuat: {len(INTERN_CACHE)} data.")
        
        # Menggunakan waktu WIB untuk penarikan cache harian
        today_str = datetime.datetime.now(AppConfig.WIB).strftime('%Y-%m-%d')
        cursor.execute("SELECT intern_name FROM attendance_logs WHERE date(absent_at) = ?", (today_str,))
        absen_tercatat = {row['intern_name'] for row in cursor.fetchall()}
        conn.close()
        print(f"✅ Cache absensi hari ini dimuat: {len(absen_tercatat)} orang.")
        
    except FileNotFoundError as e:
        print(f"🔥 KESALAHAN KRITIS: File tidak ditemukan: {e}"); exit()
    except Exception as e:
        print(f"🔥 KESALAHAN KRITIS saat startup: {e}"); exit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server memulai..."); load_all_data(); yield; print("🛑 Server berhenti.")

app = FastAPI(title="DeepFace Attendance API (Local DB)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(AppConfig.STATIC_DIR.resolve())), name="static")
app.mount("/images", StaticFiles(directory=str(AppConfig.IMAGE_STORAGE_DIR.resolve())), name="images")
templates = Jinja2Templates(directory=str(AppConfig.TEMPLATES_DIR.resolve()))

# --- Endpoint HTML ---
@app.get("/", response_class=HTMLResponse)
async def serve_main_dashboard(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/{page_name}.html", response_class=HTMLResponse)
async def serve_pages(request: Request, page_name: str):
    # Menggunakan resolve() untuk memastikan path absolut
    template_path = AppConfig.TEMPLATES_DIR.resolve() / f"{page_name}.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Halaman tidak ditemukan")
    return templates.TemplateResponse(f"{page_name}.html", {"request": request})

# --- Endpoint Absensi ---
@app.post("/recognize")
async def recognize_face(request: Request):
    try:
        frame_bytes = await request.body()
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Logika Absensi dan Pengenalan
        embedding = extract_face_features(frame)
        if embedding is None: return JSONResponse(status_code=400, content={"audio_track": "S002"})
        
        # Pastikan model dimuat sebelum digunakan
        if knn_model is None or label_encoder is None:
             raise RuntimeError("Model klasifikasi belum dimuat.")

        DISTANCE_THRESHOLD = 0.5
        distances, _ = knn_model.kneighbors(embedding)
        closest_distance = distances[0][0]
        predicted_label_id = knn_model.predict(embedding)[0]
        
        person_name = "unknown"
        if closest_distance <= DISTANCE_THRESHOLD:
            person_name = label_encoder.inverse_transform([predicted_label_id])[0]
        
        # 1. Pengecekan Duplikasi
        if person_name in absen_tercatat:
            # Gunakan f-string untuk path file yang dinamis
            temp_audio_name = f"temp_{int(time.time())}.mp3"
            temp_audio_path = AppConfig.IMAGE_STORAGE_DIR / temp_audio_name
            
            tts = gTTS(text=f"Anda Sudah Melakukan Absensi Hari Ini, {person_name}", lang='id')
            tts.save(str(temp_audio_path))
            
            return JSONResponse(status_code=200, content={"status": "fail", "message": f"DUPLIKAT: {person_name}", "audio_track": f"/images/{temp_audio_name}"})
        
        # 2. Pengecekan Tidak Dikenal
        if person_name == "unknown" or person_name not in INTERN_CACHE:
            # Mengembalikan status 404 custom (sesuai logika asli Anda)
            return JSONResponse(status_code=404, content={"audio_track": "S003"})
        
        # 3. Absen Berhasil
        intern_data = INTERN_CACHE.get(person_name)
        image_filename = f"{person_name}_{int(time.time())}.jpg"
        cv2.imwrite(str(AppConfig.IMAGE_STORAGE_DIR / image_filename), frame)
        
        conn = db_connect()
        cursor = conn.cursor()
        
        # Menggunakan WIB untuk absent_at
        absent_time_wib = datetime.datetime.now(AppConfig.WIB).isoformat()
        
        cursor.execute("INSERT INTO attendance_logs (intern_id, intern_name, universitas, kategori, image_url, absent_at) VALUES (?, ?, ?, ?, ?, ?)",
            (intern_data['id'], person_name, intern_data['universitas'], intern_data['kategori'], f"/images/{image_filename}", absent_time_wib))
            
        conn.commit(); conn.close(); absen_tercatat.add(person_name)
        
        return JSONResponse(status_code=200, content={"status": "success", "audio_track": audio_tracking.get(person_name)})
        
    except Exception as e:
        # Menambahkan respons detail pada error 500 jika tidak di lingkungan produksi
        print(f"❌ Error di /recognize: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# --- API Endpoints untuk Dashboard ---
def query_db(query, args=(), one=False):
    conn = db_connect(); cur = conn.cursor(); cur.execute(query, args)
    rv = cur.fetchall(); conn.close()
    return (dict(rv[0]) if rv else None) if one else [dict(row) for row in rv]

@app.get("/api/system-start-date")
async def get_system_start_date():
    first_log = query_db("SELECT MIN(absent_at) as start_date FROM attendance_logs", one=True)
    start_date = first_log['start_date'] if first_log and first_log['start_date'] else datetime.datetime.now(AppConfig.WIB).isoformat()
    return {"system_start_date": start_date, "current_date": datetime.datetime.now(AppConfig.WIB).isoformat()}

@app.get("/api/today-active-interns")
async def get_today_active_interns():
    # Menggunakan datetime dari pytz untuk mendapatkan tanggal hari ini sesuai WIB
    today_wib = datetime.datetime.now(AppConfig.WIB).strftime('%Y-%m-%d')
    rows = query_db("SELECT intern_name, universitas, kategori, absent_at, image_url, intern_id FROM attendance_logs WHERE date(absent_at) = ? ORDER BY absent_at DESC", [today_wib])
    active_interns = [{"name": r['intern_name'], "jobdesk": f"{r['kategori']} - {r['universitas']}", "recognition_time": r['absent_at'], "capture_image": r['image_url'], "intern_id": r['intern_id']} for r in rows]
    return {"total_active": len(active_interns), "active_interns": active_interns}

# ... (get_attendance_dates endpoint tidak diubah)

@app.get("/api/attendance-dates-with-range")
async def get_attendance_dates_with_range():
    start_row = query_db("SELECT MIN(date(absent_at)) as start_date FROM attendance_logs", one=True)
    
    if not start_row or not start_row['start_date']: return {"date_range": [], "total_dates": 0}
    
    start_date = datetime.datetime.strptime(start_row['start_date'], '%Y-%m-%d').date()
    end_date = datetime.datetime.now(AppConfig.WIB).date() # Menggunakan WIB untuk tanggal akhir
    
    attended_dates = {row['attendance_date'] for row in query_db("SELECT DISTINCT date(absent_at) as attendance_date FROM attendance_logs")}
    date_range = []; current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        date_range.append({"date": date_str, "has_attendance": date_str in attended_dates})
        current_date += datetime.timedelta(days=1)
        
    return {"date_range": date_range, "total_dates": len(attended_dates)}

# ... (endpoint API lainnya tidak diubah karena sudah benar)

@app.get("/api/attendance-summary")
async def get_attendance_summary():
    # Menggunakan datetime dari pytz untuk mendapatkan tanggal hari ini sesuai WIB
    today_wib = datetime.datetime.now(AppConfig.WIB).strftime('%Y-%m-%d')
    result = query_db("SELECT COUNT(id) as total FROM attendance_logs WHERE date(absent_at) = ?", [today_wib], one=True)
    return {"total_attendees": result['total'] if result else 0}


# --- Main Runner ---
def open_browser():
    """Fungsi yang akan dijalankan setelah jeda untuk membuka browser."""
    time.sleep(2)
    webbrowser.open_new_tab("http://127.0.0.1:8000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    
    print("✅ Server siap menerima koneksi. Browser akan terbuka secara otomatis...")
    # Mengubah "backend.main:app" menjadi objek 'app' langsung untuk mencegah masalah uvicorn reload
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)