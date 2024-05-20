from get_label_paths import process_folders
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9, min_tracking_confidence=0.7)


def extract_hand_landmarks(video_path):
    data = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data.append([landmark.x for landmark in hand_landmarks.landmark] + [landmark.y for landmark in hand_landmarks.landmark])
    cap.release()
    return data


def process_data(train, test):
    training = []
    training_labels = []
    testing = []
    testing_labels = []
    video_paths, video_paths2, labels, labels2 = process_folders(train, test)
    print(video_paths, video_paths2, labels, labels2)
    for i, video_path in enumerate(video_paths):
        hand_landmarks = extract_hand_landmarks(video_path)
        print(f"Video path: {video_path}, Number of hand landmarks: {len(hand_landmarks)}")
        training.extend(hand_landmarks)
        training_labels.extend([labels[i]] * len(hand_landmarks))

    for i, video_path in enumerate(video_paths2):
        hand_landmarks = extract_hand_landmarks(video_path)
        print(f"Video path: {video_path}, Number of hand landmarks: {len(hand_landmarks)}")
        testing.extend(hand_landmarks)
        testing_labels.extend([labels2[i]] * len(hand_landmarks))
    return np.array(training), np.array(testing), np.array(training_labels), np.array(testing_labels)



# data = data.reshape(data.shape[0], 21, 2, 1)
