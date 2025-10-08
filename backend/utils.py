from deepface import DeepFace
import cv2
import numpy as np

# ArcFace adalah salah satu model paling akurat saat ini 
# dan lebih tangguh dalam menghadapi variasi seperti kacamata.
MODEL_NAME = "ArcFace"

def extract_face_features(image_path_or_array):
    """
    Mengekstrak fitur wajah (embedding) dari sebuah gambar menggunakan DeepFace.
    Fungsi ini bisa menerima path file (string) atau gambar dalam bentuk array numpy.

    Returns:
        list: Sebuah list berisi satu embedding jika wajah terdeteksi, 
              atau None jika tidak ada wajah.
    """
    try:
        # DeepFace.represent() adalah fungsi inti untuk mengubah wajah menjadi angka.
        # 'enforce_detection=False' memastikan program tidak error jika tidak ada wajah,
        # melainkan mengembalikan hasil yang bisa kita periksa.
        embedding_objs = DeepFace.represent(
            img_path=image_path_or_array, 
            model_name=MODEL_NAME, 
            enforce_detection=False,
            detector_backend='retinaface' # Menggunakan backend deteksi yang cepat
        )

        # Hasilnya adalah list, karena satu gambar bisa punya banyak wajah.
        # Kita periksa apakah ada hasil dan apakah wajah benar-benar terdeteksi.
        if not embedding_objs or embedding_objs[0]["facial_area"]["w"] == 0:
            # print(f"Wajah tidak terdeteksi di {image_path_or_array}")
            return None

        # Kita hanya ambil embedding dari wajah pertama yang ditemukan.
        embedding = embedding_objs[0]["embedding"]

        # Kita kembalikan dalam format list agar sama seperti library sebelumnya,
        # ini membuat sisa kode kita tidak perlu banyak diubah.
        return [embedding] 

    except Exception as e:
        # Terkadang DeepFace bisa error jika file gambar rusak, dll.
        # print(f"Error saat mengekstrak fitur: {e}")
        return None

def detect_face(frame: np.ndarray):
    """
    Fungsi ini sekarang hanya sebagai placeholder. 
    DeepFace menangani deteksi wajahnya sendiri di dalam `represent()`.
    Kita tetap menyimpannya agar struktur kode kita mirip dengan sebelumnya.
    """
    return frame