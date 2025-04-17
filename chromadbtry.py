import chromadb
from chromadb import Settings
import keras_facenet
import numpy as np
import pickle
import os
from PIL import Image
import cv2
from mtcnn import MTCNN

# Setup Chroma Client with correct settings
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(name="faces")

# FaceNet model
embedder = keras_facenet.FaceNet()
detector = MTCNN()

def extract_face(image_path, required_size=(160, 160)):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    if not results:
        return None
    x1, y1, w, h = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    face = rgb[y1:y1+h, x1:x1+w]
    image = Image.fromarray(face).resize(required_size)
    return np.asarray(image)

def get_embedding(face):
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    emb = embedder.embeddings(face)
    return emb[0]

# Example: Add all images from a folder
DATA_DIR = "train_class"

for filename in os.listdir(DATA_DIR):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in ['.jpg', '.png','.jpeg']:
        continue
    image_path = os.path.join(DATA_DIR, filename)
    face = extract_face(image_path)
    if face is None:
        continue
    emb = get_embedding(face)
    collection.add(
        documents=[name],
        embeddings=[emb.tolist()],
        ids=[name]
    )



# Predict function
def predict_from_frame(frame):
    face = extract_face_from_frame(frame)
    if face is None:
        print("No face found")
        return
    emb = get_embedding(face)
    results = collection.query(query_embeddings=[emb.tolist()], n_results=1)
    
    if results['distances'][0]:
        match_id = results['ids'][0][0]
        distance = results['distances'][0][0]
        print(f"Predicted: {match_id} (Distance: {distance:.4f})")
        return match_id
    else:
        print("No match found")
        return None

def extract_face_from_frame(frame, required_size=(160, 160)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    if not results:
        return None
    x1, y1, w, h = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    face = rgb[y1:y1+h, x1:x1+w]
    image = Image.fromarray(face).resize(required_size)
    return np.asarray(image)

