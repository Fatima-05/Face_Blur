import os
os.makedirs("faces", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np

KNOWN_FACES_DIR = "faces"
PADDING = 30

# More realistic threshold for this method
DISTANCE_THRESHOLD = 1.2

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Store multiple embeddings per person
known_embeddings = {}  # {name: [emb1, emb2, ...]}

# -----------------------------
# Better embedding
# -----------------------------
def get_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    points = np.array([[p.x, p.y, p.z] for p in lm])

    # Center
    points = points - np.mean(points, axis=0)

    # Normalize scale
    norm = np.linalg.norm(points)
    if norm > 0:
        points = points / norm

    return points.flatten()  # USE ALL LANDMARKS

# -----------------------------
# Load known faces
# -----------------------------
print("Loading known faces...\n")

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = cv2.imread(path)

        if img is None:
            continue

        emb = get_embedding(img)
        if emb is not None:
            name = os.path.splitext(file)[0]

            if name not in known_embeddings:
                known_embeddings[name] = []

            known_embeddings[name].append(emb)
            print(f"✓ Loaded: {file}")
        else:
            print(f"✗ No face in: {file}")

print(f"\nPeople loaded: {len(known_embeddings)}\n")

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Face Recognition + Blur", cv2.WINDOW_NORMAL)

print("Press 's' to save | 'f' fullscreen | 'q' quit\n")

current_face = None

# -----------------------------
# Recognition function
# -----------------------------
def recognize(emb):
    best_name = "Unknown"
    best_dist = float("inf")

    for name, emb_list in known_embeddings.items():
        # Compare with all samples of that person
        dists = [np.linalg.norm(emb - e) for e in emb_list]
        avg_dist = np.mean(dists)

        if avg_dist < best_dist:
            best_dist = avg_dist
            best_name = name

    if best_dist < DISTANCE_THRESHOLD:
        return best_name, best_dist
    return "Unknown", best_dist

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(max(0, bbox.xmin * w - PADDING))
            y = int(max(0, bbox.ymin * h - PADDING))
            bw = int(min(w - x, bbox.width * w + 2 * PADDING))
            bh = int(min(h - y, bbox.height * h + 2 * PADDING))

            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size == 0:
                continue

            current_face = face_roi.copy()

            name = "Unknown"
            color = (0, 0, 255)

            emb = get_embedding(face_roi)

            if emb is not None and known_embeddings:
                name, dist = recognize(emb)

                if name != "Unknown":
                    color = (0, 255, 0)
                    print(f"Recognized: {name} ({dist:.4f})")

            # Blur unknown
            if name == "Unknown":
                blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
                display[y:y+bh, x:x+bw] = blurred

            cv2.rectangle(display, (x, y), (x+bw, y+bh), color, 3)
            cv2.putText(display, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition + Blur", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('f'):
        current = cv2.getWindowProperty("Face Recognition + Blur", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Face Recognition + Blur",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if current == cv2.WINDOW_NORMAL else cv2.WINDOW_NORMAL
        )

    elif key == ord('s') and current_face is not None:
        name_input = input("\nEnter name: ").strip()
        if name_input:
            path = os.path.join(KNOWN_FACES_DIR, f"{name_input}_{len(os.listdir(KNOWN_FACES_DIR))}.jpg")
            cv2.imwrite(path, current_face)
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