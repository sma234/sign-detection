import cv2
import numpy as np
import os

from classes import CLASSES

TEMPLATES = {}

for cls in CLASSES:
    class_dir = os.path.join("templates", cls)
    if not os.path.isdir(class_dir):
        continue

    tmpl_list = []
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(class_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        tmpl_list.append(cv2.resize(img, (32, 32)))

    if tmpl_list:
        TEMPLATES[cls] = tmpl_list

print(f"Loaded {sum(len(v) for v in TEMPLATES.values())} templates across {len(TEMPLATES)} classes.")


def classify_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))

    best_label, best_score = None, -1.0

    for label, tmpl_list in TEMPLATES.items():
        for tmpl in tmpl_list:
            score = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_label = label

    if best_score < 0.4:
        return None, best_score
    return best_label, best_score


def find_sign_candidates(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Red wraps around 0° in HSV, requiring two ranges
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   140, 120]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 140, 120]), np.array([180, 255, 255])),
    )
    mask_blue   = cv2.inRange(hsv, np.array([108, 100,  70]), np.array([125, 255, 255]))
    mask_green  = cv2.inRange(hsv, np.array([ 70,  40,  60]), np.array([ 95, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([ 20,  80,  80]), np.array([ 35, 255, 255]))

    mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_blue),
                          cv2.bitwise_or(mask_green, mask_yellow))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois, bboxes = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        # Discard elongated blobs; traffic signs are roughly compact
        if 4 * np.pi * area / (perimeter ** 2) < 0.5:
            continue

        x, y, w, h = cv2.boundingRect(c)
        pad = 10
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame.shape[1])
        y2 = min(y + h + pad, frame.shape[0])

        rois.append(frame[y1:y2, x1:x2])
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return rois, bboxes, mask


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        rois, bboxes, mask = find_sign_candidates(frame)

        for roi, (x, y, w, h) in zip(rois, bboxes):
            label, score = classify_roi(roi)
            if label is not None:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Traffic Sign Detection", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
