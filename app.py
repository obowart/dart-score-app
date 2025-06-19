# app_v3_1.py
import cv2
import numpy as np
import streamlit as st
import tempfile
import math
from PIL import Image
from arrow_detector import detect_arrows, get_sector_from_mask, get_ring_from_distance

st.set_page_config(layout="centered", page_title="Dart Score Analyzer 3.1")
st.title("🎯 Dart Score Analyzer 3.1 - Template + Sector Masking")

st.markdown("""
Gunakan kamera 1:1 untuk mengambil gambar dartboard secara lurus. 
Sistem akan mengenali arah panah menggunakan **template panah** dan menentukan skor berdasarkan **masking sektor**.
""")

camera_image = st.camera_input("📷 Ambil Gambar Dartboard")

if camera_image is not None and st.button("✔️ Proses Gambar"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(camera_image.getbuffer())
        img_path = tmp_file.name

    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 600))

    # Load template dan masker sektor
    template = cv2.imread("arrow_template.png")
    sector_mask = cv2.imread("mask_sector_20.png", cv2.IMREAD_GRAYSCALE)
    center = (300, 300)

    # Deteksi panah
    arrows = detect_arrows(img, template)
    total_score = 0
    score_log = []

    for i, arrow in enumerate(arrows):
        head = arrow["head"]
        tail = arrow["tail"]

        # Dapatkan sektor
        sector = get_sector_from_mask(head, sector_mask)
        ring_name, ring_multiplier = get_ring_from_distance(center, head)

        if sector is not None and ring_multiplier > 0:
            score = sector * ring_multiplier
        else:
            score = 0

        # Gambar arah panah
        cv2.arrowedLine(img, tail, head, (0, 255, 0), 2, tipLength=0.4)
        total_score += score
        score_log.append(f"Panah {i+1}: Head {head} → Sektor {sector} ({ring_name}) → {score} poin")

    # Tampilkan hasil
    st.image(img, caption="📸 Deteksi Panah & Sektor", channels="BGR")
    st.subheader("📋 Detail Skor:")
    for row in score_log:
        st.text(row)
    st.success(f"🎯 Total Skor Kamu: {total_score} poin")
