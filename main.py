import os
import cv2
import numpy as np

# -----------------------------
# Config
# -----------------------------
KNOWN_FACES_DIR = "faces"
PADDING = 20
STABLE_FRAMES = 5      # frames to confirm recognition
FRAME_SCALE = 0.5
ORB_FEATURES = 500
THRESHOLD = 0.55       # stricter threshold to reduce false positives

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -----------------------------
# Variables
# -----------------------------
current_face = None
last_name = "Unknown"
stable_count = 0

# -----------------------------
# Caffe DNN face detector
# -----------------------------
DETECTOR_PROTO = os.path.join(os.getcwd(), "deploy.prototxt")
DETECTOR_MODEL = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(DETECTOR_PROTO, DETECTOR_MODEL)

# -----------------------------
# ORB setup
# -----------------------------
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# -----------------------------
# OpenCV-based SSIM
# -----------------------------
def ssim_cv(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (img1.shape[1], img1.shape[0]))
    C1 = 6.5025
    C2 = 58.5225
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)
    mu1_mu2 = mu1 * mu2
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    sigma1_sq = cv2.GaussianBlur(img1**2, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# -----------------------------
# Load known faces (folder-per-person)
# -----------------------------
known_faces = {}  # name: list of dicts {orb, hist, img}

def compute_descriptors(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return des

def compute_hist(face):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

print("Loading known faces...")
for person in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person)
    if os.path.isdir(person_dir):
        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue
            des = compute_descriptors(img)
            hist = compute_hist(img)
            if person not in known_faces:
                known_faces[person] = []
            known_faces[person].append({'orb': des, 'hist': hist, 'img': img})
            print(f"Loaded: {person}/{file}")
print(f"Total known people: {len(known_faces)}\n")

# -----------------------------
# Recognition function
# -----------------------------
def recognize(face):
    des = compute_descriptors(face)
    hist = compute_hist(face)
    best_name = "Unknown"
    best_score = 0
    if des is None or not known_faces:
        return best_name

    for name, samples in known_faces.items():
        sample_scores = []
        for s in samples:
            # ORB matching
            matches = bf.match(des, s['orb']) if s['orb'] is not None else []
            orb_score = len(matches)/len(des) if matches else 0
            # Histogram correlation
            hist_score = cv2.compareHist(hist, s['hist'], cv2.HISTCMP_CORREL)
            # SSIM
            ssim_score = ssim_cv(face, s['img'])
            combined_score = 0.5*orb_score + 0.3*hist_score + 0.2*ssim_score
            sample_scores.append(combined_score)
        max_score = max(sample_scores)  # use max instead of average
        if max_score > best_score:
            best_score = max_score
            best_name = name

    if best_score < THRESHOLD:
        best_name = "Unknown"
    return best_name

# -----------------------------
# Webcam loop
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

    frame = cv2.flip(frame, 1)  # mirror camera
    display = frame.copy()

    small_frame = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    h, w = small_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence < 0.6:
            continue

        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = int(x1 / FRAME_SCALE)
        y1 = int(y1 / FRAME_SCALE)
        x2 = int(x2 / FRAME_SCALE)
        y2 = int(y2 / FRAME_SCALE)

        x1 = max(0, x1-PADDING)
        y1 = max(0, y1-PADDING)
        x2 = min(frame.shape[1], x2+PADDING)
        y2 = min(frame.shape[0], y2+PADDING)

        face_roi = frame[y1:y2, x1:x2]
        current_face = face_roi.copy()

        name = recognize(face_roi)
        color = (0,255,0) if name != "Unknown" else (0,0,255)

        # global last_name, stable_count
        if name == last_name:
            stable_count += 1
        else:
            stable_count = 0
        last_name = name

        if stable_count < STABLE_FRAMES and name != "Unknown":
            color = (0,0,255)
            name = "Unknown"

        if name == "Unknown":
            blurred = cv2.GaussianBlur(face_roi, (51,51), 30)
            display[y1:y2, x1:x2] = blurred

        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
        cv2.putText(display, name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition + Blur", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and current_face is not None:
        name_input = input("\nEnter name: ").strip()
        if name_input:
            person_dir = os.path.join(KNOWN_FACES_DIR, name_input)
            os.makedirs(person_dir, exist_ok=True)
            count = len(os.listdir(person_dir))
            path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(path, current_face)
            des = compute_descriptors(current_face)
            hist = compute_hist(current_face)
            if name_input not in known_faces:
                known_faces[name_input] = []
            known_faces[name_input].append({'orb': des, 'hist': hist, 'img': current_face})
            print(f"Saved: {name_input}/{count}.jpg")

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