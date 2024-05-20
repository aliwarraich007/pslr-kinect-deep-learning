import cv2
import numpy as np
import os

input_dir = '../../dataset/samples/train'
output_dir = '../../dataset/samples/augmented'

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
    delta_w = 700 - scaled_img.shape[1]
    delta_h = 720 - scaled_img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # Assuming the images are color; adjust for grayscale if necessary
    return cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def translate_image(img):
    rows, cols = img.shape[:2]
    trans_limit = translation_factor
    trans_x = np.random.randint(-trans_limit, trans_limit)
    trans_y = np.random.randint(-trans_limit, trans_limit)
    matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    return cv2.warpAffine(img, matrix, (cols, rows))


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
        augmentations = [adjust_brightness, add_noise, rotate_image, scale_image, translate_image]
        augmentation_names = ['brightness', 'noise', 'rotate', 'scale', 'translate']

        for aug, name in zip(augmentations, augmentation_names):
            example_frame = aug(frames[0])
            out_path = os.path.join(category_output_path, f"{name}_{video_file}")
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (700, 720))
            for frame in frames:
                augmented_frame = aug(frame)
                out.write(augmented_frame)
            out.release()

print("Augmentation complete!")
