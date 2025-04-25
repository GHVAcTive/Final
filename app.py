import streamlit as st
import os
from deepface import DeepFace
from src.preprocess import extract_frames, detect_faces_in_frames

# Setup
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🔍 Deepfake Detection App")

option = st.radio("Select file type:", ("Image", "Video"))

# Make sure upload folders exist
os.makedirs("uploaded_files", exist_ok=True)
os.makedirs("uploaded_files/frames", exist_ok=True)
os.makedirs("uploaded_files/faces", exist_ok=True)

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image_path = os.path.join("uploaded_files", "uploaded_image.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        st.image(image_path, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded successfully!")

        try:
            st.write("🧠 Analyzing the image...")
            analysis = DeepFace.analyze(img_path=image_path, actions=['emotion', 'age', 'gender'], enforce_detection=False)
            st.write("### 🤖 DeepFace Analysis:")
            st.json(analysis)
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        video_path = os.path.join("uploaded_files", "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(video_path)
        st.success("✅ Video uploaded successfully!")

        # Frame extraction
        st.write("📤 Extracting frames from video...")
        extract_frames(video_path, "uploaded_files/frames", max_frames=10)
        detect_faces_in_frames("uploaded_files/frames", "uploaded_files/faces")
        
        st.write("✅ Frames and faces saved.")
