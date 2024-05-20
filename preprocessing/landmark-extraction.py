import cv2
import mediapipe as mp
import os
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Directory paths
dataset_dir = '../dataset/processed/test'
processed_dir = '../landmarks_raw-test'


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
                data.append([landmark.x for landmark in hand_landmarks.landmark] +
                            [landmark.y for landmark in hand_landmarks.landmark])
    cap.release()
    return data


def save_landmarks_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def process_videos_in_dataset(dataset_dir, processed_dir):
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        output_folder_path = os.path.join(processed_dir, folder)
        os.makedirs(output_folder_path, exist_ok=True)
        all_video_data = []

        for file in os.listdir(folder_path):
            if file.endswith('.mp4'):
                video_path = os.path.join(folder_path, file)
                landmarks = extract_hand_landmarks(video_path)
                all_video_data.append({'video': file, 'landmarks_raw': landmarks})
                print(f"Processed {video_path}")
        output_file = os.path.join(output_folder_path, f"{folder}_all_landmarks.json")
        save_landmarks_to_json(all_video_data, output_file)
        print(f"All data for class {folder} saved to {output_file}")


process_videos_in_dataset(dataset_dir, processed_dir)
