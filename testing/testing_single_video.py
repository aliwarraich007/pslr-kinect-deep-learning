import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import json
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)


with open("../labels.json", 'r') as json_file:
    validation_classes = json.load(json_file)


validation_classes = np.unique(np.array(validation_classes))
img_height = 0.9

cnn_model = load_model('../models/cnn/psl_cnn_aug.keras')


def extract_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Extract landmarks_raw of the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        # Convert landmarks_raw to flat list
        hand_landmarks_list = [landmark.x for landmark in hand_landmarks] + [landmark.y for landmark in hand_landmarks]
        return hand_landmarks_list
    else:
        return None


def preprocess_data(hand_landmarks):
    # Convert list to numpy array and reshape to match CNN input shape
    reshapedData = np.array(hand_landmarks).reshape(-1, 21, 2, 1)
    return reshapedData


def predict_labels(data):
    probabilities = cnn_model.predict(data)
    predicted_label_index = np.argmax(probabilities)
    return predicted_label_index



video_path = "../bilal_demo/bilal_cheek.mp4"
label_encoder = LabelEncoder()
# Fit the label encoder to the original labels
label_encoder.fit(validation_classes)


def crop_video1(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_top = 0
    crop_bottom = int(frame_height * img_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = video_path.replace('.mp4', '_cropped.mp4')
    out = cv2.VideoWriter(out_path, fourcc, 15.0, (frame_width, crop_bottom - crop_top))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[crop_top:crop_bottom, :]
        out.write(cropped_frame)
    cap.release()
    out.release()

    return out_path


crop_path = crop_video1(video_path)
cap = cv2.VideoCapture(crop_path)
predicted_labels_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hand_landmarks = extract_hand_landmarks(frame)
    if hand_landmarks:
        data = preprocess_data(hand_landmarks)
        # Predict label
        predicted_label_index = predict_labels(data)
        # Decode predicted label using label encoder
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        # Append predicted label to the list
        predicted_labels_list.append(predicted_label)


cap.release()
predicted_labels_array = np.array(predicted_labels_list)
unique_labels, label_counts = np.unique(predicted_labels_array, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} occurrences")


print("Predicted class labels:", predicted_labels_array)
