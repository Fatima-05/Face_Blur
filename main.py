import os
import cv2
import numpy as np
import face_recognition
from collections import defaultdict

os.makedirs("faces", exist_ok=True)

KNOWN_FACES_DIR = "faces"
PADDING         = 30

TOLERANCE       = 0.55

PROCESS_EVERY   = 5

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


print("Loading known faces...\n")
known_embeddings = []  
known_names      = []  

for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    count = 0
    for file in sorted(os.listdir(person_dir)):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img  = cv2.imread(os.path.join(person_dir, file))
        if img is None:
            continue
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb  = np.ascontiguousarray(rgb, dtype=np.uint8)

        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        if not locations:
            locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=2)
        if not locations:
            print(f"No face found in {person_name}/{file} — skipping")
            continue
        encs = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if encs:
            known_embeddings.append(encs[0])
            known_names.append(person_name)
            count += 1
        else:
            print(f"No face found in {file} — skipping")
    if count:
        print(f"{person_name}: {count} embedding(s)")
    else:
        print(f"{person_name}: no usable photos")

print(f"{len(known_embeddings)} total embeddings loaded")
print(f"Tolerance: {TOLERANCE}  (lower=stricter)\n")
print("Controls:  S = save face   F = fullscreen   Q = quit\n")


def identify(face_img_rgb):
    """
    Returns (name, distance) or ("Unknown", None).
    Uses 128-dim face embeddings — properly distinguishes
    different people regardless of skin tone, beard, etc.
    """
    h, w = face_img_rgb.shape[:2]
    if h > 150:
        scale        = 150 / h
        face_img_rgb = cv2.resize(face_img_rgb,
                                   (int(w * scale), 150))

    encs = face_recognition.face_encodings(face_img_rgb, num_jitters=0)
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


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

WIN = "Face Recognition"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

frame_count = 0
last_results = []   
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

    if frame_count % PROCESS_EVERY == 0:
        last_results = []
        for (x, y, bw, bh) in boxes:
            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size == 0:
                continue
            rgb_roi      = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            name, dist   = identify(rgb_roi)
            color        = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            dist_str     = f" ({dist:.2f})" if dist is not None else ""
            last_results.append((x, y, bw, bh, name, color, dist_str))

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

        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            known_embeddings.append(encs[0])
            known_names.append(name_input)
            print(f"Saved + embedded photo #{count+1} for '{name_input}'\n")
        else:
            print(f"Photo saved but no face detected in it — try again\n")

cap.release()
cv2.destroyAllWindows()