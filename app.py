# app.py

import streamlit as st
import os
import cv2
from PIL import Image
from collections import Counter
from deepface import DeepFace
import tempfile
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Deepfake Detection with DeepFace", layout="centered")
st.title("🧠 Deepfake Detection using DeepFace")

option = st.radio("Select file type:", ("Image", "Video", "Evaluate"))

os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("uploaded_files/frames", exist_ok=True)

# =====================================
# IMAGE HANDLING
# =====================================
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_path = os.path.join("uploaded_files", uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(image_path, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded successfully!")

        try:
            st.write("🔍 Analyzing with DeepFace...")
            result = DeepFace.analyze(img_path=image_path, actions=["emotion", "age", "gender"], enforce_detection=False)
            st.write("🎯 **Analysis Result:**")
            st.json(result[0])
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

# =====================================
# VIDEO HANDLING
# =====================================
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        video_path = os.path.join("uploaded_files", "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(video_path)
        st.success("✅ Video uploaded successfully!")

        st.write("📤 Extracting frames...")

        cap = cv2.VideoCapture(video_path)
        frame_dir = "uploaded_files/frames"
        count = 0
        saved = 0
        success = True

        while success:
            success, frame = cap.read()
            if not success:
                break
            if count % 20 == 0:
                frame_path = os.path.join(frame_dir, f"frame_{saved}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            count += 1
        cap.release()

        st.write(f"📸 Saved {saved} frames.")

        st.write("🧪 Analyzing each frame...")

        predictions = []

        for frame_file in os.listdir(frame_dir):
            if frame_file.endswith(".jpg"):
                path = os.path.join(frame_dir, frame_file)
                try:
                    result = DeepFace.analyze(img_path=path, actions=["emotion"], enforce_detection=False)
                    dominant_emotion = result[0]["dominant_emotion"]
                    predictions.append(dominant_emotion)

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(path, width=120)
                    with col2:
                        st.write(f"**Emotion:** {dominant_emotion}")
                except Exception as e:
                    st.warning(f"Could not analyze {frame_file}: {e}")

        if predictions:
            final_emotion = Counter(predictions).most_common(1)[0][0]
            st.success(f"🎯 **Most Frequent Emotion in Video:** {final_emotion}")
        else:
            st.warning("⚠️ No faces/emotions detected in video.")

# =====================================
# EVALUATE MODEL ON MULTIPLE IMAGES
# =====================================
elif option == "Evaluate":
    uploaded_images = st.file_uploader("Upload multiple face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images and len(uploaded_images) >= 3:
        temp_dir = tempfile.mkdtemp()
        image_paths = []

        for img in uploaded_images:
            img_path = os.path.join(temp_dir, img.name)
            with open(img_path, "wb") as f:
                f.write(img.getbuffer())
            image_paths.append(img_path)

        st.write(f"📸 {len(image_paths)} images uploaded.")

        pairs = list(itertools.combinations(image_paths, 2))
        y_true, y_pred = [], []

        for img1, img2 in pairs:
            try:
                result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
                y_true.append(1 if result["verified"] else 0)
                y_pred.append(1 if result["distance"] < result["threshold"] else 0)
            except Exception as e:
                st.warning(f"⚠️ Error comparing images: {e}")

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            st.write("\n### 📊 Model Evaluation Results")
            st.write(f"- Accuracy: `{acc:.2f}`")
            st.write(f"- Precision: `{prec:.2f}`")
            st.write(f"- Recall: `{rec:.2f}`")
            st.write(f"- F1 Score: `{f1:.2f}`")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)
        else:
            st.warning("❗ Not enough data to evaluate.")
