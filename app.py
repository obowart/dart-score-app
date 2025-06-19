import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import math

SECTOR_ANGLES = [
    (351, 9, 6), (9, 27, 13), (27, 45, 4), (45, 63, 18), (63, 81, 1),
    (81, 99, 20), (99, 117, 5), (117, 135, 12), (135, 153, 9),
    (153, 171, 14), (171, 189, 11), (189, 207, 8), (207, 225, 16),
    (225, 243, 7), (243, 261, 19), (261, 279, 3), (279, 297, 17),
    (297, 315, 2), (315, 333, 15), (333, 351, 10)
]

def get_sector(angle):
    for start, end, score in SECTOR_ANGLES:
        if start > end:
            if angle >= start or angle < end:
                return score
        else:
            if start <= angle < end:
                return score
    return 0

def calculate_score(center_x, center_y, x, y):
    dx, dy = x - center_x, y - center_y
    distance = math.hypot(dx, dy)
    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
    sector_score = get_sector(angle)

    if distance <= 15:
        return 50  # Bullseye merah
    elif distance <= 40:
        return 25  # Bullseye hijau
    elif 100 <= distance <= 110:
        return sector_score * 3
    elif 160 <= distance <= 170:
        return sector_score * 2
    elif distance <= 180:
        return sector_score
    else:
        return 0

def detect_darts_and_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Gambar tidak bisa dimuat.", None, None

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

    all_contours = sorted(contours_red + contours_green, key=cv2.contourArea, reverse=True)[:5]

    score_summary = []
    total_score = 0
    center_x, center_y = 300, 300

    for i, cnt in enumerate(all_contours):
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)

            # Koreksi arah panah dari persegi panjang (mengasumsikan head berada di ujung)
            if h > w:
                head_x = x + w // 2
                head_y = y  # atas
            else:
                head_x = x
                head_y = y + h // 2  # kiri

            dx, dy = head_x - center_x, head_y - center_y
            distance = math.hypot(dx, dy)
            if distance > 180:
                continue  # Abaikan panah di luar papan dart

            score = calculate_score(center_x, center_y, head_x, head_y)
            total_score += score

            # Tampilkan kotak hanya di ujung
            cv2.rectangle(output, (head_x - 10, head_y - 10), (head_x + 10, head_y + 10), (0, 255, 0), 2)
            score_summary.append(
                f"Panah {i+1}: Head x={head_x}, y={head_y} → {score} poin"
            )

    return score_summary, output, total_score

def run_web_app():
    st.title("🎯 Dart Score Analyzer 2.1 — Koreksi Titik Head Panah")
    st.write("Ambil gambar dartboard dari kamera, lalu klik ✔️ untuk hitung skor berdasarkan titik ujung panah (head).")

    camera_image = st.camera_input("📷 Ambil Foto Dartboard")

    if camera_image is not None:
        if st.button("✔️ Proses Gambar"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(camera_image.getbuffer())
                temp_path = tmp_file.name
            scores, result_img, total = detect_darts_and_score(temp_path)
            if isinstance(scores, str):
                st.error(scores)
            else:
                st.image(result_img, caption="📸 Hasil Deteksi Panah (Head Only)", channels="BGR")
                st.subheader("📋 Hasil Deteksi Panah:")
                for score in scores:
                    st.text(score)
                st.success(f"🎯 Total Skor Kamu: {total} poin")

run_web_app()
