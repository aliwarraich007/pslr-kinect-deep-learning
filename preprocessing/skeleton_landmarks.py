import cv2
import mediapipe as mp
import os
import json

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Directory paths
dataset_dir = '../dataset/samples/train'
processed_dir = '../landmarks-skeleton-augmented/'


def extract_landmarks(video_path):
    data = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            frame_data = {}

            if results.pose_landmarks:
                frame_data['pose'] = [[landmark.x, landmark.y, landmark.z] for landmark in
                                      results.pose_landmarks.landmark]

            if results.face_landmarks:
                frame_data['face'] = [[landmark.x, landmark.y, landmark.z] for landmark in
                                      results.face_landmarks.landmark]

            if results.left_hand_landmarks:
                frame_data['left_hand'] = [[landmark.x, landmark.y, landmark.z] for landmark in
                                           results.left_hand_landmarks.landmark]

            if results.right_hand_landmarks:
                frame_data['right_hand'] = [[landmark.x, landmark.y, landmark.z] for landmark in
                                            results.right_hand_landmarks.landmark]

            data.append(frame_data)

    cap.release()
    return data


def process_videos_in_dataset(dataset_dir, processed_dir):
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        output_folder_path = os.path.join(processed_dir, folder)
        os.makedirs(output_folder_path, exist_ok=True)
        all_video_data = []

        for file in os.listdir(folder_path):
            if file.endswith('.mp4'):
                video_path = os.path.join(folder_path, file)
                landmarks = extract_landmarks(video_path)
                all_video_data.append({'video': file, 'landmarks_raw': landmarks})
                print(f"Processed {video_path}")


process_videos_in_dataset(dataset_dir, processed_dir)
