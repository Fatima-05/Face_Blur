import os
os.makedirs("faces", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

KNOWN_FACES_DIR = "faces"
PADDING = 30
DISTANCE_THRESHOLD = 0.42   # You can tune this

print("Loading known faces (multi-photo per person)...\n")

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

detector = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Store embeddings per person
person_embeddings = defaultdict(list)   # person_name → list of embeddings

def get_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    points = np.array([[p.x, p.y, p.z] for p in lm])
    points = points - points.mean(axis=0)
    points = points / (np.max(np.abs(points)) + 1e-6)
    return points.flatten()[:100]

# Load all photos from subfolders
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    count = 0
    for file in os.listdir(person_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                emb = get_embedding(img)
                if emb is not None:
                    person_embeddings[person_name].append(emb)
                    count += 1

    if count > 0:
        print(f"✓ Loaded {count} photos for '{person_name}'")
    else:
        print(f"✗ No valid faces found for '{person_name}'")

print(f"\nTotal people loaded: {len(person_embeddings)}\n")

# ========================= MAIN LOOP =========================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Face Recognition + Blur", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition + Blur", 1280, 720)

print("Press 's' to save new photo | 'f' = fullscreen | 'q' = quit\n")

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
            bw = int(min(w - x, bbox.width * w + 2*PADDING))
            bh = int(min(h - y, bbox.height * h + 2*PADDING))

            face_roi = frame[y:y+bh, x:x+bw]
            if face_roi.size == 0:
                continue

            name = "Unknown"
            color = (0, 0, 255)

            current_emb = get_embedding(face_roi)
            if current_emb is not None and person_embeddings:
                best_dist = float('inf')
                best_person = "Unknown"

                for person, emb_list in person_embeddings.items():
                    if not emb_list:
                        continue
                    # Average distance to all photos of this person
                    dists = [np.linalg.norm(current_emb - emb) for emb in emb_list]
                    avg_dist = np.mean(dists)
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_person = person

                if best_dist < DISTANCE_THRESHOLD:
                    name = best_person
                    color = (0, 255, 0)

            # Blur unknown
            if name == "Unknown":
                blurred = cv2.GaussianBlur(face_roi, (51, 51), 40)
                display[y:y+bh, x:x+bw] = blurred

            cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 4)
            cv2.putText(display, name, (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    cv2.imshow("Face Recognition + Blur", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        cv2.setWindowProperty("Face Recognition + Blur", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if cv2.getWindowProperty("Face Recognition + Blur", cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_NORMAL else cv2.WINDOW_NORMAL)
    elif key == ord('s') and results.detections:
        name_input = input("\nEnter name for this person: ").strip()
        if name_input:
            person_dir = os.path.join(KNOWN_FACES_DIR, name_input)
            os.makedirs(person_dir, exist_ok=True)
            count = len(os.listdir(person_dir))
            filename = os.path.join(person_dir, f"{name_input}_{count+1}.jpg")
            cv2.imwrite(filename, face_roi)
            print(f"✅ Saved photo #{count+1} for {name_input}")

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