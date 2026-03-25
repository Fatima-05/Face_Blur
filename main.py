import os
import cv2
import numpy as np

# -----------------------------
# Config
# -----------------------------
KNOWN_FACES_DIR = "faces"
PADDING = 20
DISTANCE_THRESHOLD = 0.6  # lower = stricter
STABLE_FRAMES = 3          # frames required to confirm recognition
FRAME_SCALE = 0.5          # downscale frame for speed

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -----------------------------
# Variables
# -----------------------------
current_face = None
last_name = "Unknown"
stable_count = 0

# -----------------------------
# Load DNN face detector
# -----------------------------
DETECTOR_PROTO = os.path.join(os.getcwd(), "deploy.prototxt")
DETECTOR_MODEL = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(DETECTOR_PROTO, DETECTOR_MODEL)

# -----------------------------
# Load known faces (multiple samples per person)
# -----------------------------
known_faces = {}  # name: list of embeddings

def get_embedding(face):
    """Resizes and flattens face into a vector for simple comparison"""
    face_resized = cv2.resize(face, (100, 100))
    vec = face_resized.flatten().astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec

print("Loading known faces...")
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = cv2.imread(path)
        if img is not None:
            emb = get_embedding(img)
            name = os.path.splitext(file)[0].split("_")[0]
            if name not in known_faces:
                known_faces[name] = []
            known_faces[name].append(emb)
            print(f"Loaded: {file}")
print(f"Total known people: {len(known_faces)}\n")

# -----------------------------
# Recognition function
# -----------------------------
def recognize(face_emb):
    best_name = "Unknown"
    best_dist = float("inf")
    for name, emb_list in known_faces.items():
        # Average distance across multiple samples per person
        dists = [np.linalg.norm(face_emb - emb) for emb in emb_list]
        avg_dist = np.mean(dists)
        if avg_dist < best_dist:
            best_dist = avg_dist
            best_name = name
    if best_dist < DISTANCE_THRESHOLD:
        return best_name, best_dist
    return "Unknown", best_dist

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow("Face Recognition + Blur", cv2.WINDOW_NORMAL)

print("Press 's' to save new face | 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Flip camera (selfie view)
    # -----------------------------
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Resize frame for faster DNN
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    h, w = small_frame.shape[:2]

    blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Scale back to original frame size
        x1 = int(x1 / FRAME_SCALE)
        y1 = int(y1 / FRAME_SCALE)
        x2 = int(x2 / FRAME_SCALE)
        y2 = int(y2 / FRAME_SCALE)

        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(frame.shape[1], x2 + PADDING)
        y2 = min(frame.shape[0], y2 + PADDING)

        face_roi = frame[y1:y2, x1:x2]
        current_face = face_roi.copy()

        face_emb = get_embedding(face_roi)
        name = "Unknown"
        color = (0, 0, 255)  # red for unknown

        if known_faces:
            detected_name, dist = recognize(face_emb)

            # Frame stability
            if detected_name == last_name:
                stable_count += 1
            else:
                stable_count = 0
            last_name = detected_name

            if stable_count >= STABLE_FRAMES:
                name = detected_name
                if name != "Unknown":
                    color = (0, 255, 0)
                    print(f"Recognized: {name} ({dist:.4f})")

        # Blur unknown faces
        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 30)
            display[y1:y2, x1:x2] = blurred

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition + Blur", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and current_face is not None:
        name_input = input("\nEnter name: ").strip()
        if name_input:
            count = len([f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(name_input)])
            path = os.path.join(KNOWN_FACES_DIR, f"{name_input}_{count}.jpg")
            cv2.imwrite(path, current_face)
            emb = get_embedding(current_face)
            if name_input not in known_faces:
                known_faces[name_input] = []
            known_faces[name_input].append(emb)
            print(f"Saved: {name_input}")

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