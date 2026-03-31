import os
import cv2
import numpy as np
from collections import defaultdict

os.makedirs("faces", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Settings ───────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "faces"
PADDING         = 30
CONFIDENCE_MIN  = 0.7
FACE_SIZE       = (200, 200)

# Multiplier over each person's own average training confidence.
# A live face must score BELOW (own_avg * THRESHOLD_MULTIPLIER) to be accepted.
# Lower = stricter. Start at 2.0, raise to 3.0 if known faces aren't recognised.
THRESHOLD_MULTIPLIER = 2.5
MIN_THRESHOLD        = 60.0   # never accept if confidence is above this anyway

# ── DNN face detector ──────────────────────────────────────────────────────────
print("Loading face detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ Detector loaded\n")

# ── LBPH recognizer ────────────────────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=16, grid_x=8, grid_y=8
)

id_to_name = {}
name_to_id = {}
faces_data  = []
labels_data = []
next_id     = [0]   # list so retrain() can mutate it


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray


def detect_faces(frame):
    h, w  = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets  = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < CONFIDENCE_MIN:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1-PADDING); y1 = max(0, y1-PADDING)
        x2 = min(w, x2+PADDING); y2 = min(h, y2+PADDING)
        boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes


# ── Load faces & train ─────────────────────────────────────────────────────────
print("Loading known faces...\n")
person_train_confs = defaultdict(list)   # name → [confidence on own training photos]

for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    pid   = next_id[0]; next_id[0] += 1
    id_to_name[pid]        = person_name
    name_to_id[person_name] = pid
    count = 0
    for file in sorted(os.listdir(person_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                faces_data.append(preprocess(img))
                labels_data.append(pid)
                count += 1
    if count:
        print(f"  ✓ {person_name}: {count} photo(s)  [id={pid}]")
    else:
        print(f"  ✗ {person_name}: no valid photos")

trained = False
person_thresholds = {}

def train_and_calibrate():
    global trained
    if not faces_data:
        return
    recognizer.train(faces_data, np.array(labels_data, dtype=np.int32))
    trained = True

    # Measure each person's confidence on their own training photos
    # LBPH will score its own training samples very low (well-fitted)
    # We use this to set a per-person acceptance ceiling
    per_person_confs = defaultdict(list)
    for face, label in zip(faces_data, labels_data):
        pred_label, conf = recognizer.predict(face)
        if pred_label == label:   # correctly identified own photo
            per_person_confs[label].append(conf)

    print("\nPer-person calibration:")
    for pid, name in id_to_name.items():
        confs = per_person_confs.get(pid, [])
        if confs:
            avg = np.mean(confs)
            std = np.std(confs) if len(confs) > 1 else avg * 0.5
            # Threshold = avg + generous margin, capped at MIN_THRESHOLD
            threshold = min(avg * THRESHOLD_MULTIPLIER + std * 2, MIN_THRESHOLD)
            threshold = max(threshold, avg + 5)   # always at least 5 above avg
        else:
            threshold = MIN_THRESHOLD
        person_thresholds[pid] = threshold
        avg_str = f"{np.mean(confs):.1f}" if confs else "N/A"
        print(f"  {name}: avg_conf={avg_str}  threshold={threshold:.1f}")

train_and_calibrate()
print(f"\n✅ Recognizer trained on {len(faces_data)} photos\n")
print("Controls:  S = save face   F = fullscreen   Q = quit\n")


# ── Webcam ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

WIN = "Face Recognition"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

last_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    display = frame.copy()
    boxes   = detect_faces(frame)
    last_boxes = boxes

    for (x, y, bw, bh) in boxes:
        face_roi = frame[y:y+bh, x:x+bw]
        if face_roi.size == 0:
            continue

        name  = "Unknown"
        color = (0, 0, 255)
        conf_display = ""

        if trained:
            gray        = preprocess(face_roi)
            label, conf = recognizer.predict(gray)
            threshold   = person_thresholds.get(label, MIN_THRESHOLD)

            if conf < threshold:
                name         = id_to_name.get(label, "Unknown")
                color        = (0, 200, 0)
                conf_display = f" ({conf:.0f})"

        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
            display[y:y+bh, x:x+bw] = blurred

        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 3)
        label_y = max(y - 35, 0)
        cv2.rectangle(display, (x, label_y), (x+bw, y), color, -1)
        cv2.putText(display, name + conf_display, (x+6, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(display, "S: save  F: fullscreen  Q: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
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

        gray = preprocess(face_roi)
        if name_input not in name_to_id:
            pid = next_id[0]; next_id[0] += 1
            id_to_name[pid]         = name_input
            name_to_id[name_input]  = pid
        faces_data.append(gray)
        labels_data.append(name_to_id[name_input])

        print(f"✅ Saved photo #{count+1} for '{name_input}' — retraining...\n")
        train_and_calibrate()
        print("✅ Done\n")

cap.release()
cv2.destroyAllWindows()