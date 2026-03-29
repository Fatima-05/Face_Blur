import os
import cv2
import numpy as np
from collections import defaultdict

os.makedirs("faces", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


KNOWN_FACES_DIR  = "faces"
PADDING          = 30
CONFIDENCE_MIN   = 0.7
FACE_SIZE        = (200, 200)


REJECTION_SIGMA  = 2.5


print("Loading face detector...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")
print("Detector loaded\n")


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
        x1 = max(0, x1 - PADDING); y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING); y2 = min(h, y2 + PADDING)
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes



print("Loading known faces...\n")


person_faces  = defaultdict(list)   

for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    count = 0
    for file in sorted(os.listdir(person_dir)):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                person_faces[person_name].append(preprocess(img))
                count += 1
    if count:
        print(f"  ✓ {person_name}: {count} photo(s)")
    else:
        print(f"  ✗ {person_name}: no valid photos")

print(f"\nTotal people: {len(person_faces)}\n")




def build_person_model(person_name, all_person_faces):
    """
    Train a recognizer where:
      label 0 = this person
      label 1 = everyone else (negatives)
    Then measure the person's own LBPH confidence distribution
    to set a personalised rejection threshold.
    """
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8
    )

    pos_faces = all_person_faces[person_name]
    neg_faces = []
    for other, faces in all_person_faces.items():
        if other != person_name:
            neg_faces.extend(faces)

    train_faces  = pos_faces + neg_faces
    train_labels = ([0] * len(pos_faces)) + ([1] * len(neg_faces))

    if not train_faces:
        return None, None, None

    rec.train(train_faces, np.array(train_labels, dtype=np.int32))

    
    own_confs = []
    for face in pos_faces:
        label, conf = rec.predict(face)
        if label == 0:   
            own_confs.append(conf)

    if not own_confs:
        own_confs = []
        for face in pos_faces:
            _, conf = rec.predict(face)
            own_confs.append(conf)

    mean = np.mean(own_confs)
    std  = np.std(own_confs) if len(own_confs) > 1 else mean * 0.3
    threshold = mean + REJECTION_SIGMA * std

    return rec, threshold, mean


print("Building per-person models...\n")
person_models = {}   

for person_name in person_faces:
    rec, threshold, mean_conf = build_person_model(person_name, person_faces)
    if rec is not None:
        person_models[person_name] = (rec, threshold)
        print(f"  {person_name}: own avg conf={mean_conf:.1f}  "
              f"rejection threshold={threshold:.1f}")

print(f"\n{len(person_models)} models ready\n")


def identify_face(gray_face):
    """
    Run the face against every person's model.
    A person is accepted only if their model predicts label=0
    AND confidence is below their personal threshold.
    Among all accepted persons, return the one with lowest confidence.
    Returns (name, confidence) or ("Unknown", None).
    """
    candidates = []
    for person_name, (rec, threshold) in person_models.items():
        label, conf = rec.predict(gray_face)
        if label == 0 and conf < threshold:
            candidates.append((person_name, conf))

    if not candidates:
        return "Unknown", None

    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def retrain_all():
    """Rebuild all models after new photos are added."""
    global person_models
    person_models = {}
    for person_name in person_faces:
        rec, threshold, mean_conf = build_person_model(person_name, person_faces)
        if rec is not None:
            person_models[person_name] = (rec, threshold)
            print(f"  {person_name}: threshold={threshold:.1f}")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

WIN = "Face Recognition"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

print("Controls:  S = save face   F = fullscreen   Q = quit\n")
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

        gray = preprocess(face_roi)
        name, conf = identify_face(gray)

        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
            display[y:y+bh, x:x+bw] = blurred
            color = (0, 0, 255)
            label = "Unknown"
        else:
            color = (0, 200, 0)
            label = f"{name} ({conf:.0f})"

        cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 3)
        label_y = max(y - 35, 0)
        cv2.rectangle(display, (x, label_y), (x+bw, y), color, -1)
        cv2.putText(display, label, (x + 6, y - 8),
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
        largest  = max(last_boxes, key=lambda b: b[2] * b[3])
        x, y, bw, bh = largest
        face_roi = frame[y:y+bh, x:x+bw]

        name_input = input("\nEnter name for this person: ").strip()
        if not name_input:
            continue

        person_dir = os.path.join(KNOWN_FACES_DIR, name_input)
        os.makedirs(person_dir, exist_ok=True)
        count    = len([f for f in os.listdir(person_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
        filename = os.path.join(person_dir, f"{name_input}_{count+1}.jpg")
        cv2.imwrite(filename, face_roi)

        person_faces[name_input].append(preprocess(face_roi))
        print(f"Saved photo #{count+1} for '{name_input}' — retraining...\n")
        retrain_all()
        print("Models updated\n")

cap.release()
cv2.destroyAllWindows()

# import os
# os.makedirs("faces", exist_ok=True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import cv2
# import mediapipe as mp
# import numpy as np

# KNOWN_FACES_DIR = "faces"
# PADDING = 30

# # Very forgiving threshold for now
# DISTANCE_THRESHOLD = 0.60

# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh

# detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# known_embeddings = []
# known_names = []

# def get_embedding(image):
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)
#     if not results.multi_face_landmarks:
#         return None
#     lm = results.multi_face_landmarks[0].landmark
#     points = np.array([[p.x, p.y, p.z] for p in lm])
#     points = points - points.mean(axis=0)
#     points = points / (np.max(np.abs(points)) + 1e-6)
#     return points.flatten()[:100]

# # Load known faces
# print("Loading known faces...\n")
# for file in os.listdir(KNOWN_FACES_DIR):
#     if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#         img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
#         if img is not None:
#             emb = get_embedding(img)
#             if emb is not None:
#                 known_embeddings.append(emb)
#                 known_names.append(os.path.splitext(file)[0])
#                 print(f"✓ Loaded: {file}")
#             else:
#                 print(f"✗ No face in: {file}")

# print(f"\nTotal known faces: {len(known_names)}\n")

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# cv2.namedWindow("Face Recognition + Blur", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Face Recognition + Blur", 1280, 720)

# print("Press 's' to save new face | 'f' = fullscreen | 'q' = quit")
# print("Show the saved faces to the camera and see if name appears.\n")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     display = frame.copy()

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = detector.process(rgb)

#     if results.detections:
#         for det in results.detections:
#             bbox = det.location_data.relative_bounding_box
#             h, w, _ = frame.shape

#             x = int(max(0, bbox.xmin * w - PADDING))
#             y = int(max(0, bbox.ymin * h - PADDING))
#             bw = int(min(w - x, bbox.width * w + 2*PADDING))
#             bh = int(min(h - y, bbox.height * h + 2*PADDING))

#             face_roi = frame[y:y+bh, x:x+bw]
#             if face_roi.size == 0:
#                 continue

#             name = "Unknown"
#             color = (0, 0, 255)

#             emb = get_embedding(face_roi)
#             if emb is not None and known_embeddings:
#                 distances = [np.linalg.norm(emb - k) for k in known_embeddings]
#                 min_dist = min(distances)
#                 if min_dist < DISTANCE_THRESHOLD:
#                     idx = np.argmin(distances)
#                     name = known_names[idx]
#                     color = (0, 255, 0)
#                     print(f"Recognized: {name} (distance = {min_dist:.4f})")   # Debug info

#             if name == "Unknown":
#                 blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
#                 display[y:y+bh, x:x+bw] = blurred

#             cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 4)
#             cv2.putText(display, name, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

#     cv2.imshow("Face Recognition + Blur", display)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('f'):
#         cv2.setWindowProperty("Face Recognition + Blur", cv2.WND_PROP_FULLSCREEN, 
#                               cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("Face Recognition + Blur", cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_NORMAL else cv2.WINDOW_NORMAL)
#     elif key == ord('s') and results.detections:
#         name_input = input("\nEnter name: ").strip()
#         if name_input:
#             cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name_input}.jpg"), face_roi)
#             print(f"Saved {name_input}")

# cap.release()
# cv2.destroyAllWindows()