import cv2
import requests
import time
import os
import pygame
import tempfile

SERVER_URL_BASE = "http://127.0.0.1:8000"
RECOGNIZE_URL = f"{SERVER_URL_BASE}/recognize"
LOCAL_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'backend', 'generated_audio')

def play_audio(track_info: str):
    if not track_info: return
    try:
        # Periksa apakah track_info adalah URL dari server (diawali dengan '/')
        if track_info.startswith('/'):
            # Ini adalah audio dinamis, kita perlu mengunduhnya
            audio_url = f"{SERVER_URL_BASE}{track_info}"
            print(f"🔊 Mengunduh audio dinamis dari {audio_url}...")
            
            response = requests.get(audio_url, stream=True)
            response.raise_for_status() # Akan error jika gagal diunduh
            
            # Simpan ke file temporer
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            # Putar file temporer
            pygame.mixer.music.load(tmp_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): # Tunggu sampai selesai
                pygame.time.Clock().tick(10)
            
            # Hapus file temporer setelah selesai diputar
            os.remove(tmp_file_path)

        else:
            # Ini adalah audio lokal yang sudah ada
            file_path = os.path.join(LOCAL_AUDIO_DIR, f"{track_info}.mp3")
            if os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                print(f"🔊 Memutar audio lokal: {track_info}.mp3")
            else:
                print(f"⚠️ File audio lokal tidak ditemukan: {track_info}.mp3")

    except Exception as e:
        print(f"❌ Error saat memutar audio: {e}")

def run_webcam_attendance():
    pygame.init()
    pygame.mixer.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Tidak bisa membuka kamera.")
        return
    print("✅ Kamera siap. Tekan 'SPASI' untuk absen, 'Q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow('Webcam Absensi (DeepFace)', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            print("\n📸 Mengambil gambar...")
            _, image_bytes = cv2.imencode('.jpg', frame)
            
            print("✈️  Mengirim gambar ke server...")
            try:
                response = requests.post(RECOGNIZE_URL, data=image_bytes.tobytes(), headers={'Content-Type': 'image/jpeg'}, timeout=17)
                
                result = response.json()
                print(f"💬 Server: {result.get('message', 'N/A')}")
                play_audio(result.get('audio_track'))

            except requests.exceptions.RequestException as e:
                print(f"❌ Gagal terhubung ke server: {e}")
            
            time.sleep(1) # Beri jeda singkat
            print("\n✅ Kamera siap kembali...")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    run_webcam_attendance()