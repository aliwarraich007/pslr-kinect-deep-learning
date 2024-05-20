import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from joblib import load
from collections import deque


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)

with open('svc_model_aug.pkl', 'rb') as file:
    svc_model = pickle.load(file)
scaler = load('../../models/svc/scaler_aug.joblib')


def extract_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        # Pad with zeros if only one hand is detected
        if len(results.multi_hand_landmarks) == 1:
            hand_landmarks_list.extend([0] * 63)
        return hand_landmarks_list
    else:
        return None

# Function to preprocess hand landmarks data
def preprocess_data(hand_landmarks_buffer):
    # Convert buffer to numpy array
    data = np.array(hand_landmarks_buffer).reshape(len(hand_landmarks_buffer), -1, 63)

    # Extract features (original data, velocities, accelerations)
    velocities = np.diff(data, axis=0)
    accelerations = np.diff(velocities, axis=0)

    # Use only the most recent frame's data, velocity, and acceleration
    flattened_data = data[-1].flatten() if len(data) > 0 else np.zeros(63)
    flattened_velocities = velocities[-1].flatten() if len(velocities) > 0 else np.zeros(63)
    flattened_accelerations = accelerations[-1].flatten() if len(accelerations) > 0 else np.zeros(63)

    features = np.hstack((flattened_data, flattened_velocities, flattened_accelerations))

    # Ensure the features have the correct size
    expected_feature_size = scaler.n_features_in_
    if features.shape[0] < expected_feature_size:
        features = np.pad(features, (0, expected_feature_size - features.shape[0]), mode='constant')
    elif features.shape[0] > expected_feature_size:
        features = features[:expected_feature_size]

    # Scale the features
    scaled_features = scaler.transform([features])
    return scaled_features

# Load classes
with open('../../labels.json', 'r') as f:
    class_data = json.load(f)
    classes = class_data['classes']

# Initialize buffer to store previous frames' landmarks
buffer_size = 3  # You can adjust this size
hand_landmarks_buffer = deque(maxlen=buffer_size)

# Capture video from the laptop camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract hand landmarks from the frame
    hand_landmarks = extract_hand_landmarks(frame)

    if hand_landmarks:
        # Ensure hand_landmarks have the correct number of features
        if len(hand_landmarks) == 126:  # 2 hands x 21 landmarks x 3 coordinates (x, y, z)
            # Append to buffer
            hand_landmarks_buffer.append(hand_landmarks)

            # Only process if we have enough frames in the buffer
            if len(hand_landmarks_buffer) == buffer_size:
                # Preprocess the hand landmarks data
                data = preprocess_data(hand_landmarks_buffer)
                # Predict label
                predicted_label = svc_model.predict(data)[0]
                # Draw predicted label on the frame
                cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
