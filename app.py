import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

def detect_darts_and_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Gambar tidak bisa dimuat.", None

    img = cv2.resize(img, (600, 600))
    output = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    score_summary = []
    dart_count = 0

    for cnt in contours_red + contours_green:
        area = cv2.contourArea(cnt)
        if area > 50:
            dart_count += 1
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 2)
            score_summary.append(f"Panah {dart_count}: Lokasi ~ x={x}, y={y} (deteksi warna)")

    return score_summary, output

def run_web_app():
    st.title("🎯 Sistem Skor Otomatis Dart")
    st.write("Ambil gambar dartboard langsung dari kamera, lalu klik tombol ✔️ untuk deteksi panah.")

    camera_image = st.camera_input("📷 Ambil Foto Dartboard")

    if camera_image is not None:
        if st.button("✔️ Proses Gambar"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(camera_image.getbuffer())
                temp_path = tmp_file.name
            scores, result_img = detect_darts_and_score(temp_path)
            if isinstance(scores, str):
                st.error(scores)
            else:
                st.image(result_img, caption="📸 Hasil Deteksi Panah", channels="BGR")
                st.subheader("📋 Hasil Deteksi Panah:")
                for score in scores:
                    st.text(score)

run_web_app()
