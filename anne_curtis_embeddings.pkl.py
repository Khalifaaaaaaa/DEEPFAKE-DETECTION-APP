from keras_facenet import FaceNet
import cv2
import numpy as np
import pickle
import os
from mtcnn import MTCNN  # Use MTCNN for face detection

# Load models
embedder = FaceNet()
detector = MTCNN()  # Load MTCNN detector

def resize_with_padding(image, target_size=(160, 160)):
    """Resize an image while maintaining aspect ratio and adding padding."""
    h, w, _ = image.shape
    target_w, target_h = target_size

    # Compute scale to fit within the target size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a black background
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute padding offsets (center the resized image)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Place resized image onto the black background
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    return padded_image

def get_face_embedding(image_path):
    """Detects the face using MTCNN, crops it, resizes with padding, and extracts FaceNet embeddings."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Image at path {image_path} not found.")

    # Convert image to RGB (MTCNN requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(image_rgb)
    
    if not faces:
        raise ValueError(f"❌ No face detected in {image_path}.")

    # Get the largest detected face
    faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    x, y, w, h = faces[0]['box']

    # Ensure positive coordinates
    x, y = max(0, x), max(0, y)

    # Crop the face
    face = image[y:y+h, x:x+w]

    # Resize while maintaining aspect ratio
    resized_face = resize_with_padding(face, (160, 160))

    # Expand dimensions for model input
    resized_face = np.expand_dims(resized_face, axis=0)

    # Get embedding
    embedding = embedder.embeddings(resized_face)

    return embedding[0]  # Return 512-d embedding

# Define your image folder
downloads_path = r"C:\Users\Nat\Downloads"

# Dictionary to hold celebrity names and their image paths
celebrities = {
    "Anne Curtis": [
        "annecurtis.jpg", 
        "annecurtis1.jpg", 
        "annecurtis2.jpg", 
        "annecurtis4.jpg", 
        "annecurtis5.jpg"
    ],
    "Andrea Brillantes": [
        "andreab.jpg", 
         "andreab1.jpeg", 
          "andreab2.jpg", 
           "andreab3.jpg", 
    ],

    "Bini Mikha": [
        "mikha.jpg", 
          
          "mikha2.jpg", 
           "mikha3.jpg", 
           "mikha4.jfif", 
    ],

    "BongBong Marcos": [
        "bbm.jpg", 
          "bbm2.jpg", 
           "bbm3.jpg", 
           "bbm4.jpg", 
           "bbm5.jpg",
    ]
}

# Convert filenames to absolute paths
face_embeddings = {}

for name, image_files in celebrities.items():
    for file in image_files:
        file_path = os.path.join(downloads_path, file)
        try:
            emb = get_face_embedding(file_path)
            face_embeddings[name] = emb
            print(f"✅ {name}: Processed {file} with embedding shape {emb.shape}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# Save embeddings
with open("anne_curtis_embeddings.pkl", "wb") as f:
    pickle.dump(face_embeddings, f)

print("✅ Face embeddings saved to celebrity_face_embeddings.pkl")
