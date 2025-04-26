# app.py
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. ======== Model Loading ========
MODEL_PATH = "deepfake_detector.h5"
deepfake_model = load_model(MODEL_PATH)

# 2. ======== Configuration ========
st.set_page_config(page_title="DeepGuard Pro", layout="centered")
st.title("🔍 Professional Deepfake Detector")

# 3. ======== Core Functions ========
def preprocess_img(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_deepfake(img_path):
    try:
        processed_img = preprocess_img(img_path)
        prediction = deepfake_model.predict(processed_img)[0][0]
        label = "Real" if prediction < 0.5 else "Deepfake"
        confidence = (1 - prediction)*100 if label=="Real" else prediction*100
        return label, round(confidence, 2)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

# 4. ======== Image Analysis ========
def analyze_image(img_path):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_path, use_column_width=True)
    
    with col2:
        label, confidence = predict_deepfake(img_path)
        
        if label:
            st.success("## Analysis Results")
            st.metric("Deepfake Detection", 
                     f"{label}", 
                     f"{confidence}% Confidence")
            
            try:
                analysis = DeepFace.analyze(
                    img_path=img_path,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False
                )[0]
                
                st.markdown("### Facial Attributes")
                st.write(f"**Age:** {analysis['age']} years")
                st.write(f"**Gender:** {analysis['dominant_gender']}")
                st.write(f"**Emotion:** {analysis['dominant_emotion']}")
                
            except Exception as e:
                st.warning(f"Face analysis limited: {str(e)}")

# 5. ======== Video Analysis ========
def analyze_video(video_path):
    st.warning("Video processing may take 2-3 minutes...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 30  # Process 1 frame per second for 30fps video
    
    predictions = []
    processed = 0
    
    with st.progress(0) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if processed % frame_skip == 0:
                frame_path = f"uploaded_files/frames/frame_{processed}.jpg"
                cv2.imwrite(frame_path, frame)
                
                label, _ = predict_deepfake(frame_path)
                predictions.append(label)
                pbar.progress(min(processed/total_frames, 1.0))
            
            processed += 1
    
    if predictions:
        final = max(set(predictions), key=predictions.count)
        st.success(f"Video Analysis Result: {final}")
    else:
        st.error("No faces detected in video")

# 6. ======== Main Interface ========
def main():
    option = st.sidebar.selectbox("Select Mode", ["Image", "Video"])
    
    if option == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            analyze_image(img_path)
            
    elif option == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if uploaded_file:
            video_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            analyze_video(video_path)

if __name__ == "__main__":
    main()