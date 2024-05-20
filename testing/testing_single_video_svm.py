import cv2
import mediapipe as mp
import numpy as np
import pickle
from joblib import load

# Load the trained model and scaler
with open("../models/svc/psl_augmented.pkl", 'rb') as model_file:
    svc_model = pickle.load(model_file)

scaler = load('../models/svc/scaler_augmented.joblib')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_hand_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()
    return None

def extract_features(data, num_frames):
    if len(data) < num_frames:
        while len(data) < num_frames:
            data.append(np.zeros_like(data[0]))
    data = np.array(data[:num_frames])
    velocities = np.diff(data, axis=1)
    accelerations = np.diff(velocities, axis=1)
    flattened_data = data.reshape(data.shape[0], -1)
    flattened_velocities = velocities.reshape(velocities.shape[0], -1)
    flattened_accelerations = accelerations.reshape(accelerations.shape[0], -1)
    combined_features = np.hstack((flattened_data, flattened_velocities, flattened_accelerations))
    return combined_features

def process_video(video_path, hands, num_frames):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_hand_landmarks(frame, hands)
        if landmarks is not None:
            landmarks_list.append(landmarks)
    cap.release()
    features = extract_features(landmarks_list, num_frames)
    return features

# Load the demo video and process it
demo_video_path = "../demo/abdomen.mp4"

# Determine the expected number of frames and features
num_frames = 30
n_features = 41  # Adjust this to match the training data

# Process the video
demo_features = process_video(demo_video_path, hands, num_frames)

if demo_features is not None:
    print(f"Demo features shape before scaling: {demo_features.shape}")
    # Ensure demo_features shape matches (1, n_features) expected by scaler
    demo_features_reshaped = demo_features.reshape(1, -1)
    print(f"Demo features shape after reshaping: {demo_features_reshaped.shape}")
    demo_features_scaled = scaler.transform(demo_features_reshaped)  # Adjusting input shape for the scaler
    predicted_label = svc_model.predict(demo_features_scaled)
    print("Predicted Label:", predicted_label[0])
else:
    print("No hand landmarks detected in the video.")
