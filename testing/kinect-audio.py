from libkinect2 import Kinect2
from libkinect2.utils import depth_map_to_image
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import json
import pyttsx3
import threading

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9, min_tracking_confidence=0.6)

cnn_model = load_model('../models/cnn/psl_cnn_aug_improved.keras')

try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing text-to-speech engine: {e}")
    exit()

def extract_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        hand_landmarks_list = [landmark.x for landmark in hand_landmarks] + [landmark.y for landmark in hand_landmarks]
        return hand_landmarks_list
    else:
        return None

def preprocess_data(hand_landmarks):
    data = np.array(hand_landmarks).reshape(-1, 21, 2, 1)
    return data

def predict_labels(data):
    probabilities = cnn_model.predict(data)
    predicted_label_index = np.argmax(probabilities)
    return predicted_label_index

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

with open('../labels.json', 'r') as f:
    class_data = json.load(f)
    classes = class_data['classes']

label_encoder = LabelEncoder()
label_encoder.fit(classes)

kinect = Kinect2(use_sensors=['color', 'depth'])
kinect.connect()
kinect.wait_for_worker()

previous_label = None

for _, color_img, depth_map in kinect.iter_frames():
    hand_landmarks = extract_hand_landmarks(color_img)

    if hand_landmarks:
        data = preprocess_data(hand_landmarks)
        predicted_label_index = predict_labels(data)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        if predicted_label != previous_label:
            threading.Thread(target=speak_text, args=(predicted_label,)).start()
            previous_label = predicted_label

        cv2.putText(color_img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Sign Detection', color_img)
    cv2.imshow('Depth', depth_map_to_image(depth_map))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

kinect.disconnect()
cv2.destroyAllWindows()
