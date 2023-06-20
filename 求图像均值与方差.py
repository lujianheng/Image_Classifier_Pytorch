import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 指定数据集的路径
dataset_path = "new_dataset/0"

# 获取所有文件
files = os.listdir(dataset_path)

# 初始化变量
mean = np.zeros(3)
std = np.zeros(3)
n_images = 0

# 遍历所有文件
for file in tqdm(files):
    # 打开图像文件
    image = Image.open(os.path.join(dataset_path, file))
    # 将图像转换为numpy数组
    image = np.array(image) / 255.0  # 将像素值缩放到[0,1]范围
    # 计算图像的平均值和方差
    mean += image.mean(axis=(0, 1))
    std += image.std(axis=(0, 1))
    n_images += 1

# 计算所有图像的平均值和方差
mean /= n_images
std /= n_images

print(f"transforms.Normalize({list(mean)}, {list(std)})")
