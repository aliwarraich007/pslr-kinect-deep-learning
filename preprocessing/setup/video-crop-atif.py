import cv2


def trim_video(video_path, output_path, duration=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


for i, video_path in enumerate(video_paths):
    output_path = f"trimmed_video_{i}.mp4"  # Output file name
    trim_video(video_path, output_path, duration=4)


    def crop_video(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        crop_top = 0
        crop_bottom = int(frame_height * 0.8)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = video_path.replace('.mp4', '_cropped.mp4')
        out = cv2.VideoWriter(out_path, fourcc, 30.0, (frame_width, crop_bottom - crop_top))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the frame
            cropped_frame = frame[crop_top:crop_bottom, :]

            # Write the cropped frame to the output video file
            out.write(cropped_frame)

        # Release everything
        cap.release()
        out.release()

        return out_path


    # Crop all videos and store their updated paths
    cropped_video_paths = [crop_video(video_path) for video_path in new_video_paths]

    print("Cropped videos:", cropped_video_paths)




