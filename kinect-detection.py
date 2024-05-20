import cv2
import os
import json
from libkinect2 import Kinect2

# Initialize Kinect2 with the body sensor
kinect = Kinect2(use_sensors=['body'])
kinect.connect()
kinect.wait_for_worker()

# Directory paths
video_path = './dataset/raw/abdomen/775-abdomen.mp4'
output_file = './output/landmarks-kinect-test.json'


def extract_skeleton_landmarks():
    """
    Fetches the latest skeleton data from Kinect.
    """
    skeleton_data = {
        'pose_landmarks': [],
        'left_hand_landmarks': [],
        'right_hand_landmarks': []
    }
    for _, bodies in kinect.iter_frames():
        if len(bodies) > 0:
            body = bodies[0]  # Assuming the first detected body
            pose_landmarks = []
            left_hand_landmarks = []
            right_hand_landmarks = []

            for joint_type, joint in body['joints'].items():
                x, y, z = joint['colorX'], joint['colorY'], joint['cameraZ']
                if joint_type in [7, 8, 9, 10, 11, 12]:  # Example joint types for left hand
                    left_hand_landmarks.append([x, y, z])
                elif joint_type in [21, 22, 23, 24, 25, 26]:  # Example joint types for right hand
                    right_hand_landmarks.append([x, y, z])
                else:
                    pose_landmarks.append([x, y, z])

            skeleton_data['pose_landmarks'] = pose_landmarks
            skeleton_data['left_hand_landmarks'] = left_hand_landmarks
            skeleton_data['right_hand_landmarks'] = right_hand_landmarks

            return skeleton_data  # Return the first detected skeleton data
    return None


def process_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    all_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        skeleton = extract_skeleton_landmarks()
        if skeleton:
            all_data.append(skeleton)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Skeleton data saved to {output_file}")


# Process a single video
process_video(video_path, output_file)

kinect.disconnect()
