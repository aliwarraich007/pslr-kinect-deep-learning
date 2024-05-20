import shutil
import os


src_dir = '../../dataset/samples/train'
dest_dir = '../../dataset/samples/augmented'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".mp4"):
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, src_dir)
            dest_file_dir = os.path.join(dest_dir, relative_path)
            if not os.path.exists(dest_file_dir):
                os.makedirs(dest_file_dir)
            dest_file_path = os.path.join(dest_file_dir, file)
            shutil.copy2(src_file_path, dest_file_path)

print("All videos have been copied to the train directory.")
