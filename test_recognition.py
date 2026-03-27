import cv2
import mediapipe as mp
import numpy as np
import os

KNOWN_FACES_DIR = "faces"

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)

def get_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    points = np.array([[p.x, p.y, p.z] for p in lm])
    points = points - points.mean(axis=0)
    points = points / (np.max(np.abs(points)) + 1e-6)
    return points.flatten()[:100]

# Load known embeddings
known_embeddings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
        if img is not None:
            emb = get_embedding(img)
            if emb is not None:
                known_embeddings.append(emb)
                known_names.append(os.path.splitext(file)[0])
                print(f"Loaded: {file}")

print(f"\nLoaded {len(known_names)} known faces.\n")

cap = cv2.VideoCapture(0)
print("Show a face to the camera. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if results.multi_face_landmarks:
        emb = get_embedding(frame)
        if emb is not None and known_embeddings:
            distances = [np.linalg.norm(emb - k) for k in known_embeddings]
            min_dist = min(distances)
            best_name = known_names[np.argmin(distances)]

            if min_dist < 0.50:
                text = f"Recognized: {best_name} ({min_dist:.3f})"
                color = (0, 255, 0)
            else:
                text = f"Unknown ({min_dist:.3f})"
                color = (0, 0, 255)

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()