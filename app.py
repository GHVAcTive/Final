# app.py

import streamlit as st
import os
import cv2
from PIL import Image
from collections import Counter
from deepface import DeepFace
import tempfile

st.set_page_config(page_title="Deepfake Detection with DeepFace", layout="centered")
st.title("🧠 Deepfake Detection using DeepFace")

option = st.radio("Select file type:", ("Image", "Video"))

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

        # Extract every N-th frame
        cap = cv2.VideoCapture(video_path)
        frame_dir = "uploaded_files/frames"
        count = 0
        saved = 0
        success = True

        while success:
            success, frame = cap.read()
            if not success:
                break
            if count % 20 == 0:  # change frame interval if needed
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
