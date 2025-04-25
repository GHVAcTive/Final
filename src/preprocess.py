import cv2
import os
from PIL import Image

def extract_frames(video_path, output_folder, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames:
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(output_folder, f"frame{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()

def detect_faces_in_frames(frame_folder, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for filename in os.listdir(frame_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(frame_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                face_path = os.path.join(output_folder, f"{filename[:-4]}_face{i}.jpg")
                cv2.imwrite(face_path, face)
