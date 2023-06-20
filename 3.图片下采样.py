import os
import shutil
import numpy as np

# 对0和2进行了下采样

# 指定数据集的路径
dataset_path = "./dataset/2"
new_dataset_path = "./new_dataset/2"

# 获取所有图片的路径
all_image_paths = [os.path.join(dataset_path, image) for image in os.listdir(dataset_path)]

# 随机抽取指定数量的图片
selected_image_paths = np.random.choice(all_image_paths, 2443, replace=False)

# 创建新的文件夹
os.makedirs(new_dataset_path, exist_ok=True)

# 将选定的图片复制到新的文件夹
for image_path in selected_image_paths:
    shutil.copy(image_path, os.path.join(new_dataset_path, os.path.basename(image_path)))
