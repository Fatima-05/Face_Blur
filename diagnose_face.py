import os
import cv2
import numpy as np
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

KNOWN_FACES_DIR   = "faces"
PADDING           = 30
CONFIDENCE_MIN    = 0.7
FACE_SIZE         = (100, 100)

# ── Load detector ──────────────────────────────────────────────────────────────
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")

# ── Embedding ──────────────────────────────────────────────────────────────────
def get_embedding(face_img):
    gray    = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE)
    resized = cv2.equalizeHist(resized)
    lbp     = np.zeros_like(resized, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(resized, dy, axis=0), dx, axis=1)
        lbp     = (lbp << 1) | (shifted >= resized).astype(np.uint8)
    h, w = lbp.shape
    hist = []
    for r in range(4):
        for c in range(4):
            cell      = lbp[r*h//4:(r+1)*h//4, c*w//4:(c+1)*w//4]
            ch, _     = np.histogram(cell.flatten(), bins=256, range=(0,256))
            ch        = ch.astype(np.float32)
            ch       /= (ch.sum() + 1e-6)
            hist.extend(ch)
    return np.array(hist, dtype=np.float32)

def compare(e1, e2):
    diff = e1 - e2
    summ = e1 + e2 + 1e-6
    return float(np.sum(diff**2 / summ))

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                  (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    dets  = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < CONFIDENCE_MIN:
            continue
        box = dets[0,0,i,3:7] * np.array([w,h,w,h])
        x1,y1,x2,y2 = box.astype(int)
        x1 = max(0, x1-PADDING); y1 = max(0, y1-PADDING)
        x2 = min(w, x2+PADDING); y2 = min(h, y2+PADDING)
        boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes

# ── Load known faces ───────────────────────────────────────────────────────────
person_embeddings = defaultdict(list)
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                emb = get_embedding(img)
                person_embeddings[person_name].append(emb)

print(f"Loaded {len(person_embeddings)} people: {list(person_embeddings.keys())}")
print("\nLook at the distances printed below.")
print("Your SIMILARITY_THRESH in main.py should be set ABOVE the known-face")
print("distances and BELOW the unknown-face distances.\n")
print("Press Q to quit.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    boxes = detect_faces(frame)

    for (x, y, bw, bh) in boxes:
        face_roi = frame[y:y+bh, x:x+bw]
        if face_roi.size == 0:
            continue

        emb = get_embedding(face_roi)

        best_dist   = float('inf')
        best_person = "Unknown"
        all_dists   = {}

        for person, emb_list in person_embeddings.items():
            dists    = [compare(emb, e) for e in emb_list]
            avg_dist = float(np.mean(dists))
            all_dists[person] = avg_dist
            if avg_dist < best_dist:
                best_dist   = avg_dist
                best_person = person

        # Print distances to console every frame
        dist_str = "  ".join([f"{p}: {d:.4f}" for p, d in all_dists.items()])
        print(f"Best → {best_person} ({best_dist:.4f})   |   All: {dist_str}")

        # Show on frame
        label = f"{best_person} ({best_dist:.3f})"
        cv2.rectangle(display, (x,y), (x+bw, y+bh), (0,200,255), 2)
        cv2.putText(display, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    cv2.imshow("Diagnostics — Q to quit", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()