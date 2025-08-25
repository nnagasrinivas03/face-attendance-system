import cv2
import os
import numpy as np
from PIL import Image


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    print("Error: Haar Cascade file not loaded! Check the file path.")
    exit()
else:
    print("Haar Cascade loaded successfully.")
def face_cropped(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]
def generate_dataset():
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0
    if not os.path.exists("data"):
        os.makedirs("data")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture image")
            break
        face = face_cropped(frame)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)
        if cv2.waitKey(1) == 13 or img_id == 100:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Thank you for your cooperation. \n You have been successfully registered")
def train_classifier(data_dir="data"):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    if not image_paths:
        print("ERROR : No training images found!!!")
        return
    faces = []
    ids = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        imageNp = np.array(img, 'uint8')
        try:
            id = int(os.path.split(image_path)[1].split(".")[1])
        except ValueError:
            print(f"Skipping invalid file: {image_path}")
            continue
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    print("Traininng of the model is completed.")
def draw_boundary(img, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_faces = len(features)
    print(f"Detected {detected_faces} face(s)")
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        face = gray_img[y:y+h, x:x+w]
        id, pred = clf.predict(face)
        confidence = int(100 * (1 - pred / 300))
        if confidence > 75:
            names = {1: "Santhosh", 2: "GURUTEJA", 3: "RISHK", 4: "AKSHITH", 5: "GOWRI"}
            name = names.get(id, "UNKNOWN")
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    return img
def recognize_faces():
    cap = cv2.VideoCapture(0)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    ret, frame = cap.read()
    if not ret:
        print("ERROR: accessing the camera")
        return
    frame = draw_boundary(frame, clf)
    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
while True:
    print("\n DIGITAL ATTENDANCE RECORD\n provide your response here\n")
    print("1. new? register here please")
    print("2. Trying to snapshots me")
    print("3. Recognize me ")
    print("4. Exit")
    choice = input("please Enter your action: ")
    if choice == "1":
        print("look at the camera!!!")
        generate_dataset()
    elif choice == "2":
        train_classifier()
    elif choice == "3":
        recognize_faces()
    elif choice == "4":
        print("your attendance is been register\n Have a nice Day")
        break
    else:
        print("Invalid choice! Please enter a valid option.")
