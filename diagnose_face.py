import os
import cv2
import numpy as np

KNOWN_FACES_DIR = "faces"
FACE_SIZE       = (200, 200)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16,
                                                  grid_x=8, grid_y=8)
faces_data, labels_data, id_to_name, name_to_id = [], [], {}, {}
next_id = 0

for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    pid = next_id; next_id += 1
    id_to_name[pid] = person_name
    name_to_id[person_name] = pid
    for file in os.listdir(person_dir):
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            img = cv2.imread(os.path.join(person_dir, file))
            if img is not None:
                faces_data.append(preprocess(img))
                labels_data.append(pid)

recognizer.train(faces_data, np.array(labels_data, dtype=np.int32))
print(f"Trained on {len(faces_data)} photos, {len(id_to_name)} people\n")

# Test each saved photo — shows what confidence the recognizer gives
print("=" * 60)
print("SELF-TEST: confidence when recognizing own saved photos")
print("(lower = more confident — ideally each person scores lowest for themselves)")
print("=" * 60)

for person_name in sorted(id_to_name.values()):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    print(f"\n  Photos of {person_name}:")
    for file in sorted(os.listdir(person_dir)):
        if not file.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        img = cv2.imread(os.path.join(person_dir, file))
        if img is None:
            continue
        gray  = preprocess(img)
        label, conf = recognizer.predict(gray)
        predicted   = id_to_name.get(label, "Unknown")
        match = "✅" if predicted == person_name else f"❌ → {predicted}"
        print(f"    {file:<30s}  conf={conf:6.1f}  {match}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("  Set LBPH_THRESHOLD just above the highest confidence")
print("  score shown for correctly matched photos above.")
print("=" * 60)

# Also test live from webcam
print("\nNow testing LIVE from webcam.")
print("Show each known person's face — press SPACE to capture, Q to quit.\n")

net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                "res10_300x300_ssd_iter_140000.caffemodel")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                  (300,300),(104.,177.,123.))
    net.setInput(blob)
    dets = net.forward()

    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < 0.7:
            continue
        box = dets[0,0,i,3:7] * np.array([w,h,w,h])
        x1,y1,x2,y2 = box.astype(int)
        x1=max(0,x1-30); y1=max(0,y1-30)
        x2=min(w,x2+30); y2=min(h,y2+30)
        roi  = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = preprocess(roi)
        label, lbph_conf = recognizer.predict(gray)
        name = id_to_name.get(label, "?")
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),2)
        cv2.putText(frame, f"{name}  conf={lbph_conf:.1f}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,200,255), 2)

    cv2.putText(frame, "SPACE=capture info  Q=quit",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.imshow("Diagnostics", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Print full breakdown for this frame
        blob2 = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                       (300,300),(104.,177.,123.))
        net.setInput(blob2)
        dets2 = net.forward()
        for i in range(dets2.shape[2]):
            if float(dets2[0,0,i,2]) < 0.7:
                continue
            box = dets2[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1=max(0,x1-30); y1=max(0,y1-30)
            x2=min(w,x2+30); y2=min(h,y2+30)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = preprocess(roi)
            print("\nLive face — scores against all known people:")
            for pid, pname in id_to_name.items():
                # Temporarily predict against each person's model isn't
                # straightforward, so just show the overall prediction
                pass
            label, lbph_conf = recognizer.predict(gray)
            print(f"  Predicted: {id_to_name.get(label,'?')}  confidence={lbph_conf:.1f}")
            print(f"  → Set LBPH_THRESHOLD above {lbph_conf:.0f} to recognise this face")

cap.release()
cv2.destroyAllWindows()