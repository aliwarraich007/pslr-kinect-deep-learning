from libkinect2 import Kinect2
from libkinect2.utils import depth_map_to_image
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import json

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9,
                       min_tracking_confidence=0.6)

# Load the trained CNN model
cnn_model = load_model('./with-camera-testing/psl_lstm_aug.keras')

# Function to extract hand landmarks from a frame
def extract_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Extract landmarks of the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        # Convert landmarks to flat list
        hand_landmarks_list = [landmark.x for landmark in hand_landmarks] + [landmark.y for landmark in hand_landmarks]
        return hand_landmarks_list
    else:
        return None

# Function to preprocess hand landmarks data
def preprocess_data(hand_landmarks):
    # Convert list to numpy array and reshape to match CNN input shape
    data = np.array(hand_landmarks).reshape(-1, 21, 2, 1)
    return data

# Function to predict labels using the CNN model
def predict_labels(data):
    # Predict probabilities for each class
    probabilities = cnn_model.predict(data)
    # Get the index of the class with highest probability
    predicted_label_index = np.argmax(probabilities)
    return predicted_label_index

# Load class names from JSON file
with open('../labels.json', 'r') as f:
    class_data = json.load(f)
    classes = class_data['classes']

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit the label encoder to the original labels
label_encoder.fit(classes)

# Initialize Kinect
kinect = Kinect2(use_sensors=['color', 'depth'])
kinect.connect()
kinect.wait_for_worker()

# Main loop to capture and process frames from Kinect
for _, color_img, depth_map in kinect.iter_frames():
    # Extract hand landmarks from the frame
    hand_landmarks = extract_hand_landmarks(color_img)

    if hand_landmarks:
        # Preprocess the hand landmarks data
        data = preprocess_data(hand_landmarks)
        # Predict label
        predicted_label_index = predict_labels(data)
        # Decode predicted label using label encoder
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        # Draw predicted label on the frame
        cv2.putText(color_img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Sign Detection', color_img)
    cv2.imshow('Depth', depth_map_to_image(depth_map))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

kinect.disconnect()
cv2.destroyAllWindows()
