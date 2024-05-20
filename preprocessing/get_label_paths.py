import os


def get_video_info(root_folder):
    video_paths = []
    labels = []

    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(folder, file))
                label = os.path.basename(folder)
                labels.append(label)
    return video_paths, labels


def process_folders(root_folder, root_folder2):
    video_paths, labels = get_video_info(root_folder)
    video_paths2, labels2 = get_video_info(root_folder2)
    return video_paths, video_paths2, labels, labels2
