import cv2
import time
import pickle
import keras_facenet
import numpy as np
from PIL import Image
from mtcnn import MTCNN





def get_model_files():
    with open('svm_classifier.pkl', 'rb') as f:
        model, in_encoder, out_encoder = pickle.load(f)

        return model,in_encoder,out_encoder


def get_embeddings(face):
    embedder = keras_facenet.FaceNet()

    face = face.astype('float32') / 255.0

    # Expand dims to match model input: (1, 160, 160, 3)
    face = np.expand_dims(face, axis=0)
    embeddings = embedder.embeddings(face)
    return embeddings[0]  


def get_name(emb):
    model,in_encoder,out_encoder = get_model_files()
    emb = in_encoder.transform([emb])
    yhat_class = model.predict(emb)
    yhat_prob = model.predict_proba(emb)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_name = out_encoder.inverse_transform([class_index])
    print(f'Predicted: {predict_name[0]} ({class_probability:.2f}%)')
    return predict_name[0]

def extract_face(frame, required_size=(160, 160)):
    # Convert to RGB as OpenCV reads in BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    results = detector.detect_faces(rgb_frame)
    if len(results) == 0:
        return None

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = rgb_frame[y1:y2, x1:x2]

    # Resize to FaceNet input size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    return face_array




def start():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Starting camera... You will capture the frame after 3 seconds.")

    start_time = time.time()
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show the live camera feed
        cv2.imshow('Live Feed', frame)

        # Check if 3 seconds have passed
        if time.time() - start_time >= 3 and captured_frame is None:
            captured_frame = frame.copy()
            print('frame captured')
            cv2.imwrite("images/captured.jpg", captured_frame)
            face = extract_face(captured_frame)
            
            print("face captured")
            cv2.imwrite("images/face.jpg", face)
            emb = get_embeddings(face)
            print("Embeddings captured")
            print(get_name(emb))
            print("Frame captured after 3 seconds!")
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if captured_frame is not None:
        cv2.imshow("Captured Frame", captured_frame)
        print("Press any key to exit.")
        
        cv2.waitKey(0)


if __name__=="__main__":
    start()