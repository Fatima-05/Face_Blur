import os
import cv2
import numpy as np
import face_recognition
from collections import defaultdict

os.makedirs("faces", exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "faces"
PADDING         = 30

# Euclidean distance threshold for face embeddings.
# 0.6 is the standard value — lower = stricter.
# Lower to 0.5 if strangers are being recognised as known.
# Raise to 0.65 if known faces aren't being recognised.
TOLERANCE       = 0.55

# Process every Nth frame for speed (1 = every frame, 2 = every other, etc.)
PROCESS_EVERY   = 2

# ── Load DNN face detector ─────────────────────────────────────────────────────
# Still use OpenCV DNN for fast detection, face_recognition for embeddings
print("Loading face detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ Detector loaded\n")


def detect_faces_dnn(frame):
    h, w  = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets  = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < 0.7:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - PADDING); y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING); y2 = min(h, y2 + PADDING)
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ── Load known faces & compute embeddings ──────────────────────────────────────
print("Loading known faces...\n")
known_embeddings = []   # list of 128-dim vectors
known_names      = []   # parallel list of names

for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    count = 0
    for file in sorted(os.listdir(person_dir)):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = face_recognition.load_image_file(
            os.path.join(person_dir, file))
        encs = face_recognition.face_encodings(img)
        if encs:
            known_embeddings.append(encs[0])
            known_names.append(person_name)
            count += 1
        else:
            print(f"  ⚠️  No face found in {file} — skipping")
    if count:
        print(f"  ✓ {person_name}: {count} embedding(s)")
    else:
        print(f"  ✗ {person_name}: no usable photos")

print(f"\n✅ {len(known_embeddings)} total embeddings loaded")
print(f"Tolerance: {TOLERANCE}  (lower=stricter)\n")
print("Controls:  S = save face   F = fullscreen   Q = quit\n")


def identify(face_img_rgb):
    """
    Returns (name, distance) or ("Unknown", None).
    Uses 128-dim face embeddings — properly distinguishes
    different people regardless of skin tone, beard, etc.
    """
    encs = face_recognition.face_encodings(face_img_rgb)
    if not encs:
        return "Unknown", None

    enc = encs[0]
    if not known_embeddings:
        return "Unknown", None

    distances = face_recognition.face_distance(known_embeddings, enc)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist < TOLERANCE:
        return known_names[best_idx], best_dist
    return "Unknown", best_dist


# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

WIN = "Face Recognition"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

frame_count = 0
last_results = []   # (x, y, bw, bh, name, color)
last_boxes   = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    display    = frame.copy()
    frame_count += 1

    boxes = detect_faces_dnn(frame)
    last_boxes = boxes

    # Only run recognition every PROCESS_EVERY frames for speed
    if frame_count % PROCESS_EVERY == 0:
        last_results = []
        for (x, y, bw, bh) in boxes:
            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size == 0:
                continue
            # face_recognition expects RGB
            rgb_roi      = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            name, dist   = identify(rgb_roi)
            color        = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            dist_str     = f" ({dist:.2f})" if dist is not None else ""
            last_results.append((x, y, bw, bh, name, color, dist_str))

    # Draw results
    for (x, y, bw, bh, name, color, dist_str) in last_results:
        face_roi = frame[y:y+bh, x:x+bw]
        if face_roi.size == 0:
            continue

        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
            display[y:y+bh, x:x+bw] = blurred

        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 3)
        label_y = max(y - 35, 0)
        cv2.rectangle(display, (x, label_y), (x+bw, y), color, -1)
        cv2.putText(display, name + dist_str, (x+6, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(display, f"Tolerance={TOLERANCE}  S:save  F:fullscreen  Q:quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow(WIN, display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('f'):
        cur = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN
                              if cur == cv2.WINDOW_NORMAL
                              else cv2.WINDOW_NORMAL)

    elif key == ord('s') and last_boxes:
        largest      = max(last_boxes, key=lambda b: b[2]*b[3])
        x, y, bw, bh = largest
        face_roi     = frame[y:y+bh, x:x+bw]

        name_input = input("\nEnter name for this person: ").strip()
        if not name_input:
            continue

        person_dir = os.path.join(KNOWN_FACES_DIR, name_input)
        os.makedirs(person_dir, exist_ok=True)
        count    = len([f for f in os.listdir(person_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
        filename = os.path.join(person_dir, f"{name_input}_{count+1}.jpg")
        cv2.imwrite(filename, face_roi)

        # Compute and store embedding immediately
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            known_embeddings.append(encs[0])
            known_names.append(name_input)
            print(f"✅ Saved + embedded photo #{count+1} for '{name_input}'\n")
        else:
            print(f"⚠️  Photo saved but no face detected in it — try again\n")

cap.release()
cv2.destroyAllWindows()