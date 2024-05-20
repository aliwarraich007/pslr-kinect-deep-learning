import cv2
import numpy as np
import os

input_dir = '../../dataset/raw'
output_dir = '../dataset/augmented-mirrored'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rotation_angle = 10
scale_percent = 75
translation_factor = 10
alpha = 1.5
beta = 5
noise_level = 0.05


def adjust_brightness(img):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img):
    noise = np.random.normal(loc=0, scale=noise_level, size=img.shape)
    img_noisy = img + noise * 255
    return np.clip(img_noisy, 0, 255).astype('uint8')


def rotate_image(img):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))


def scale_image(img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    scaled_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # Normalize the size back to 480x480
    return cv2.resize(scaled_img, (480, 480), interpolation=cv2.INTER_AREA)


def translate_image(img):
    matrix = np.float32([[1, 0, translation_factor], [0, 1, translation_factor]])
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))


def mirror_frame(img):
    return cv2.flip(img, 1)


augmentations = [adjust_brightness, add_noise, rotate_image, scale_image, translate_image, mirror_frame]
augmentation_names = ['brightness', 'noise', 'rotate', 'scale', 'translate', 'mirror']

for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)
    category_output_path = os.path.join(output_dir, category)
    if not os.path.exists(category_output_path):
        os.makedirs(category_output_path)

    for video_file in os.listdir(category_path):
        video_path = os.path.join(category_path, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Save the original video
        out_path_original = os.path.join(category_output_path, f"original_{video_file}")
        out_original = cv2.VideoWriter(out_path_original, cv2.VideoWriter_fourcc(*'H264'), fps, (480, 480))
        for frame in frames:
            out_original.write(frame)
        out_original.release()

        for aug, name in zip(augmentations, augmentation_names):
            out_path = os.path.join(category_output_path, f"{name}_{video_file}")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'H264'), fps, (480, 480))
            for frame in frames:
                augmented_frame = aug(frame)
                out.write(augmented_frame)
            out.release()

            # If the augmentation is not mirror, apply it again to the mirrored frames
            if aug is not mirror_frame:
                out_path_mirror = os.path.join(category_output_path, f"{name}_mirror_{video_file}")
                out_mirror = cv2.VideoWriter(out_path_mirror, cv2.VideoWriter_fourcc(*'avc1'), fps, (480, 480))
                for frame in frames:
                    augmented_frame = aug(mirror_frame(frame))
                    out_mirror.write(augmented_frame)
                out_mirror.release()

print("Augmentation complete!")
