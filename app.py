# arrow_detector.py
import cv2
import numpy as np
import math

def detect_arrows(img, template):
    """
    Deteksi panah dengan template matching. Mengembalikan list posisi head, tail, dan bounding box.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)

    h, w = gray_template.shape[:2]
    arrows = []
    for pt in zip(*loc[::-1]):
        x, y = pt
        box = (x, y, w, h)

        # Head default di pojok kiri atas (x, y)
        head = (x, y)
        tail = (x + w, y + h)

        # Koreksi orientasi: deteksi sudut dominan dalam bounding box (versi sederhana)
        arrow_crop = gray_img[y:y+h, x:x+w]
        edges = cv2.Canny(arrow_crop, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        if lines is not None:
            longest = max(lines, key=lambda l: np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]))[0]
            hx, hy = longest[0] + x, longest[1] + y
            tx, ty = longest[2] + x, longest[3] + y
            if np.hypot(hx - x, hy - y) > np.hypot(tx - x, ty - y):
                head, tail = (tx, ty), (hx, hy)
            else:
                head, tail = (hx, hy), (tx, ty)

        arrows.append({"box": box, "head": head, "tail": tail})

    return arrows

def get_sector_from_mask(head_point, sector_mask):
    """
    Menentukan sektor berdasarkan masking segitiga radial yang sudah disiapkan sebelumnya.
    """
    if sector_mask is None:
        return None
    val = sector_mask[head_point[1], head_point[0]]
    return int(val) if val > 0 else None

def get_ring_from_distance(center, head):
    dx, dy = head[0] - center[0], head[1] - center[1]
    dist = math.hypot(dx, dy)
    if dist <= 15:
        return "bullseye", 50
    elif dist <= 40:
        return "bull", 25
    elif 100 <= dist <= 110:
        return "triple", 3
    elif 160 <= dist <= 170:
        return "double", 2
    elif dist <= 180:
        return "normal", 1
    else:
        return "miss", 0
