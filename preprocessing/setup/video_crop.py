import os
from moviepy.editor import VideoFileClip

def split_and_save(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    class_name = os.path.basename(os.path.dirname(video_path))
    clip = VideoFileClip(video_path)
    if clip.duration > 1:
        clip = clip.subclip(0.5, clip.duration - 0.5)
    normalized_clip = clip.resize((1280, 720))
    clip = normalized_clip.crop(x1=100, y1=0, x2=800, y2=normalized_clip.size[1])
    frames_per_part = 90
    duration_per_part = frames_per_part / clip.fps

    # split video from half
    center_time = clip.duration / 2
    sub_clip_1 = clip.subclip(0, min(center_time, duration_per_part))
    sub_clip_2 = clip.subclip(center_time, min(center_time + duration_per_part, clip.duration))

    # Save to train directory
    train_output_dir = os.path.join(output_dir, "train", class_name)
    os.makedirs(train_output_dir, exist_ok=True)
    train_output_file = os.path.join(train_output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_1.mp4")
    sub_clip_1.write_videofile(train_output_file, codec="libx264")

    # Save the second half to test directory
    validation_output_dir = os.path.join(output_dir, "test", class_name)
    os.makedirs(validation_output_dir, exist_ok=True)
    validation_output_file = os.path.join(validation_output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_2.mp4")
    sub_clip_2.write_videofile(validation_output_file, codec="libx264")

    clip.close()


def process_dataset(dataset_dir, output_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                split_and_save(video_path, output_dir)


dataset_dir = "../../dataset/raw"
output_dir = "../../dataset/samples/"
process_dataset(dataset_dir, output_dir)
