import cv2
import numpy as np
import glob
import os


def simple_stitched_mask_generator(input_folder, output_folder):
    """
    简易版本：假设拼接图像的背景是纯黑色
    """
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有图片
    image_paths = glob.glob(os.path.join(input_folder, "*.[pj][np]g"))
    image_paths += glob.glob(os.path.join(input_folder, "*.bmp"))
    image_paths += glob.glob(os.path.join(input_folder, "*.tiff"))

    for img_path in image_paths:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 方法1：直接检查黑色像素（最快速）
        # 如果背景是纯黑 (0,0,0)，任何非零像素都是有内容的
        if len(img.shape) == 3:
            # 对于彩色图像，检查三个通道是否都为0
            non_black_mask = np.any(img > 0, axis=2)
        else:
            # 对于灰度图像
            non_black_mask = img > 0

        # 转换为0-255范围
        mask = non_black_mask.astype(np.uint8) * 255

        # 保存掩码
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{name_without_ext}_mask.png")

        cv2.imwrite(output_path, mask)
        print(f"Generated: {filename} -> {mask.shape[1]}x{mask.shape[0]}")


# 使用
INPUT_FOLDER = "../test_data/input"
OUTPUT_FOLDER = "../test_data/mask"
simple_stitched_mask_generator(INPUT_FOLDER, OUTPUT_FOLDER)