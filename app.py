import streamlit as st
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from PIL import Image, ExifTags
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.xception import Xception, preprocess_input
import os
import tempfile
import requests

# Load FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()

# Load known face embeddings
with open("anne_curtis_embeddings.pkl", "rb") as f:
    face_embeddings = pickle.load(f)

# Load deepfake detection model
deepfake_model = load_model("deepfake_detector.h5")

# Load Xception for feature extraction
feature_extractor = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Streamlit UI
st.title("🇵🇭 Filipino Celebrity Deepfake Detection App")
st.write("Upload an **image or video** to check if it contains a known Filipino celebrity **and** whether it's a deepfake. We also score credibility based on quality and metadata.")

# Face embedding and recognition
def get_face_embedding(face_img):
    face_resized = cv2.resize(face_img, (160, 160))
    face_resized = np.expand_dims(face_resized, axis=0)
    return embedder.embeddings(face_resized)[0]

def recognize_face(face_img, threshold=0.7):
    embedding = get_face_embedding(face_img)
    best_match, best_similarity = "Unknown", 0
    for celeb_name, stored_embedding in face_embeddings.items():
        similarity = 1 - cosine(embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = celeb_name
    return (best_match, best_similarity) if best_similarity >= threshold else ("Unknown", 0)

# Deepfake detection
def detect_deepfake(pil_image):
    image = pil_image.resize((224, 224)).convert("RGB")
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    features = feature_extractor.predict(image_array)
    features = np.expand_dims(features, axis=1)
    prediction = deepfake_model.predict(features)[0][0]
    label = "Fake" if prediction >= 0.5 else "Real"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, confidence

# EXIF metadata score
def check_exif_metadata(image):
    try:
        exif_data = image._getexif()
        if exif_data:
            return 1  # metadata present
    except:
        pass
    return 0  # no metadata

# Blur score
def check_blur(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(score / 1000, 1.0)

# SerpApi web credibility
def get_credibility_from_web(query):
    api_key = "4c7ed08bd8df0d59df4386da95d2d91d2f60cc18"
    url = f"https://serpapi.com/search?q={query}&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "organic_results" in data:
            credibility_score = min(len(data["organic_results"]) / 10, 1.0)
            return credibility_score
    return 0

# Combined credibility score
def calculate_credibility_score(image, celeb_name):
    image_np = np.array(image)
    blur_score = check_blur(image_np)
    metadata_score = check_exif_metadata(image)
    web_credibility_score = get_credibility_from_web(celeb_name)
    total_score = (blur_score + metadata_score + web_credibility_score) / 3
    return total_score

# Face recognition pipeline
def process_image(image):
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return "No face detected.", None, None
    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = image_np[y:y+h, x:x+w]
        name, similarity = recognize_face(face_img)
        results.append(f"Detected: {name} (Similarity: {similarity:.2f})")
    return "\n".join(results), name, similarity

# Video frame processing (200 frames)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        return "Could not read video.", None, None, None, None

    frame_indices = np.linspace(0, frame_count - 1, 200, dtype=int)
    celeb_detected = {}
    deepfake_results = []
    credibility_scores = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        recognition_result, celeb_name, similarity = process_image(image)
        label, confidence = detect_deepfake(image)
        credibility_score = calculate_credibility_score(image, celeb_name if celeb_name else "Unknown")

        if celeb_name:
            celeb_detected[celeb_name] = celeb_detected.get(celeb_name, 0) + 1
        deepfake_results.append((label, confidence))
        credibility_scores.append(credibility_score)

    cap.release()

    most_common_celeb = max(celeb_detected, key=celeb_detected.get) if celeb_detected else "Unknown"
    avg_deepfake = np.mean([c for _, c in deepfake_results]) if deepfake_results else 0
    deepfake_label = "Fake" if avg_deepfake >= 0.5 else "Real"
    deepfake_conf = avg_deepfake if avg_deepfake >= 0.5 else 1 - avg_deepfake
    avg_cred_score = np.mean(credibility_scores) if credibility_scores else 0

    return most_common_celeb, deepfake_label, deepfake_conf, avg_cred_score, len(deepfake_results)

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file:
    file_type = uploaded_file.type
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "video" in file_type:
        st.video(tmp_path)
        with st.spinner("Processing video... this may take a moment."):
            celeb_name, label, confidence, credibility_score, frame_count = process_video(tmp_path)

            st.subheader("Celebrity Recognition")
            st.write(f"Most Detected: **{celeb_name}** (in {frame_count} frames)")

            st.subheader("Deepfake Detection")
            st.write(f"Status: **{label}**")
            st.write(f"Confidence: **{confidence:.2f}**")

            st.subheader("Video Credibility Score")
            st.write(f"Score: **{credibility_score * 100:.2f}%**")
            st.progress(credibility_score)
    else:
        image = Image.open(tmp_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image..."):
            recognition_result, celeb_name, similarity = process_image(image)
            st.subheader("Celebrity Recognition")
            st.write(recognition_result)

            label, confidence = detect_deepfake(image)
            st.subheader("Deepfake Detection")
            st.write(f"Status: **{label}**")
            st.write(f"Confidence: **{confidence:.2f}**")

            credibility_score = calculate_credibility_score(image, celeb_name)
            st.subheader("Image Credibility Score")
            st.write(f"Score: **{credibility_score * 100:.2f}%**")
            st.progress(credibility_score)
