import cv2
import numpy as np
import math

def detect_arrows(image, template, threshold=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    w, h = template.shape[1], template.shape[0]
    arrows = []
    used = []

    for pt in zip(*loc[::-1]):
        # Hindari duplikasi deteksi
        if any([abs(pt[0]-u[0]) < 20 and abs(pt[1]-u[1]) < 20 for u in used]):
            continue
        used.append(pt)

        tail = (pt[0] + w // 8, pt[1] + h // 2)
        head = (pt[0] + w - 10, pt[1] + h // 2)

        arrows.append({"tail": tail, "head": head})

    return arrows

def get_sector_from_mask(point, mask):
    if point[0] < 0 or point[1] < 0 or point[1] >= mask.shape[0] or point[0] >= mask.shape[1]:
        return None
    return int(mask[point[1], point[0]])

def get_ring_from_distance(center, point):
    distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)

    if distance <= 15:
        return ("Bullseye", 50)
    elif distance <= 35:
        return ("Outer Bull", 25)
    elif 90 <= distance <= 105:
        return ("Triple Ring", 3)
    elif 165 <= distance <= 180:
        return ("Double Ring", 2)
    elif distance > 180:
        return ("Miss", 0)
    else:
        return ("Single", 1)
