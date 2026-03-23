import cv2
import os
import glob
import numpy as np
import random
import cv2

def generate_white_masks_for_folder(image_folder, mask_save_folder):
    print('Generating white masks for folder {}'.format(image_folder))
    """
    为文件夹中的所有图片生成纯白mask

    Args:
        image_folder: 原始图片文件夹路径
        mask_save_folder: mask保存文件夹路径
    """
    # 创建保存目录
    os.makedirs(mask_save_folder, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))



    # 为每张图片生成白色mask
    for image_path in image_paths:
        # 读取图片获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            continue

        height, width = img.shape[:2]

        # 创建纯白mask（255表示白色）
        white_mask = 255 * np.ones((height, width), dtype=np.uint8)

        # 生成mask文件名
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        mask_path = os.path.join(mask_save_folder, mask_filename)

        # 保存mask
        cv2.imwrite(mask_path, white_mask)
        print(f"已生成mask: {mask_path} (尺寸: {width}x{height})")


# 使用示例
if __name__ == "__main__":
    image_folder = "../test_data/input"  # 你的图片文件夹
    mask_save_folder = "../test_data/mask"  # mask保存文件夹

    generate_white_masks_for_folder(image_folder, mask_save_folder)