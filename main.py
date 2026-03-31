import os
import cv2
import numpy as np

os.makedirs("faces", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Settings ───────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "faces"
PADDING         = 30
CONFIDENCE_MIN  = 0.7
FACE_SIZE       = (200, 200)

# LBPH confidence: 0 = perfect match, higher = worse.
# Face is accepted as known only if confidence is BELOW this number.
# Raise to 90-100 if known faces aren't recognised.
# Lower to 50-60 if strangers are being recognised as known.
LBPH_THRESHOLD  = 80.0

# ── DNN face detector ──────────────────────────────────────────────────────────
print("Loading face detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")
print("✅ Detector loaded\n")

# ── LBPH recognizer ────────────────────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=16, grid_x=8, grid_y=8
)

id_to_name  = {}
name_to_id  = {}
faces_data  = []
labels_data = []
next_id     = [0]
trained     = False


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


def retrain():
    global trained
    if not faces_data:
        return
    recognizer.train(faces_data, np.array(labels_data, dtype=np.int32))
    trained = True
    print(f"✅ Trained on {len(faces_data)} photos, {len(id_to_name)} people\n")


# ── Load known faces ───────────────────────────────────────────────────────────
print("Loading known faces...\n")
for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    pid = next_id[0]; next_id[0] += 1
    id_to_name[pid]         = person_name
    name_to_id[person_name] = pid
    count = 0
    for file in sorted(os.listdir(person_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                faces_data.append(preprocess(img))
                labels_data.append(pid)
                count += 1
    print(f"  ✓ {person_name}: {count} photo(s)  [id={pid}]")

retrain()
print(f"Threshold: {LBPH_THRESHOLD}  (confidence shown on screen — tune this value)")
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
        label_text = "Unknown"

        if trained:
            gray        = preprocess(face_roi)
            label, conf = recognizer.predict(gray)
            # Always show confidence so you can tune the threshold
            print(f"  conf={conf:.1f}  predicted={id_to_name.get(label,'?')}")

            if conf < LBPH_THRESHOLD:
                name       = id_to_name.get(label, "Unknown")
                color      = (0, 200, 0)
                label_text = f"{name} ({conf:.0f})"

        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
            display[y:y+bh, x:x+bw] = blurred

        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 3)
        label_y = max(y - 35, 0)
        cv2.rectangle(display, (x, label_y), (x+bw, y), color, -1)
        cv2.putText(display, label_text, (x+6, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(display, f"Threshold={LBPH_THRESHOLD}  S:save  F:fullscreen  Q:quit",
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

        if name_input not in name_to_id:
            pid = next_id[0]; next_id[0] += 1
            id_to_name[pid]         = name_input
            name_to_id[name_input]  = pid

        faces_data.append(preprocess(face_roi))
        labels_data.append(name_to_id[name_input])
        print(f"✅ Saved photo #{count+1} for '{name_input}' — retraining...")
        retrain()

cap.release()
cv2.destroyAllWindows()