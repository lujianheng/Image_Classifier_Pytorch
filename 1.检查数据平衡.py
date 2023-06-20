import os
import matplotlib.pyplot as plt

# 指定数据集的路径
dataset_path = "dataset"
# dataset_path = "new_dataset"

# 获取所有子文件夹的名称
# subfolders = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
subfolders = [str(i) for i in range(5)]

# 计算每个子文件夹中的文件数量
file_counts = [len(os.listdir(os.path.join(dataset_path, subfolder))) for subfolder in subfolders]

# 使用matplotlib的柱状图将结果可视化
plt.bar(subfolders, file_counts)

# 在柱状图顶部显示数量
for i, v in enumerate(file_counts):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.xlabel('Subfolder')
plt.ylabel('Number of files')
plt.title('Number of files in each subfolder')
plt.show()
# print(subfolders,file_counts)

# import os
# import matplotlib.pyplot as plt
#
# # 指定数据集的路径
# # dataset_path = "dataset"
# dataset_path = "new_dataset"
#
# # 指定横坐标的名称
# subfolders_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
#
# # 获取所有子文件夹的名称
# subfolders = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
#
# # 计算每个子文件夹中的文件数量
# file_counts = [len(os.listdir(os.path.join(dataset_path, subfolder))) for subfolder in subfolders]
#
# # 使用matplotlib的柱状图将结果可视化
# plt.bar(subfolders_names, file_counts)
#
# # 在柱状图顶部显示数量
# for i, v in enumerate(file_counts):
#     plt.text(i, v + 0.5, str(v), ha='center')
#
# plt.xlabel('Subfolder')
# plt.ylabel('Number of files')
# plt.title('Number of files in each subfolder')
# plt.show()
