import os
import numpy as np

# 指定数据集的路径
dataset_path = "dataset"

# 获取所有子文件夹的名称
subfolders = [f.name for f in os.scandir(dataset_path) if f.is_dir()]

# 计算每个子文件夹中的文件数量
file_counts = [len(os.listdir(os.path.join(dataset_path, subfolder))) for subfolder in subfolders]

# 计算文件数量的中值
median_file_count = np.median(file_counts)

print(f"The median number of files in each subfolder is {median_file_count}")
