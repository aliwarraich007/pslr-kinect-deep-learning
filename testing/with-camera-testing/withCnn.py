import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import json
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9,
                       min_tracking_confidence=0.6)
cnn_model = load_model('../../models/rnn/psl_lstm_aug.keras')


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
    # Convert list to numpy array and reshape to match CNN input shape
    data = np.array(hand_landmarks).reshape(-1, 21, 2, 1)
    return data


def predict_labels(data):
    # Predict probabilities for each class
    probabilities = cnn_model.predict(data)
    # Get the index of the class with highest probability
    predicted_label_index = np.argmax(probabilities)
    return predicted_label_index


with open('../../labels.json', 'r') as f:
    class_data = json.load(f)
    classes = class_data['classes']


label_encoder = LabelEncoder()
label_encoder.fit(classes)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hand_landmarks = extract_hand_landmarks(frame)

    if hand_landmarks:
        data = preprocess_data(hand_landmarks)
        predicted_label_index = predict_labels(data)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
