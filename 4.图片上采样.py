import os
from PIL import Image
from torchvision import transforms

# 对3和4号进行了上采样

# 指定数据集的路径
dataset_path = "./dataset/4"
new_dataset_path = "./new_dataset/4"

# 定义数据增强的转换
data_transforms = {
    'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
    'vertical_flip': transforms.RandomVerticalFlip(p=1.0),
    'rotation': transforms.RandomRotation(90)
}

# 获取所有图片的路径
all_image_paths = [os.path.join(dataset_path, image) for image in os.listdir(dataset_path)]

# 创建新的文件夹
os.makedirs(new_dataset_path, exist_ok=True)

# 对每张图片进行数据增强
for image_path in all_image_paths:
    image = Image.open(image_path)
    for transform_name, transform in data_transforms.items():
        transformed_image = transform(image)
        # 保存增强后的图片
        new_file_name = f"{os.path.basename(image_path).rsplit('.', 1)[0]}_{transform_name}.{os.path.basename(image_path).rsplit('.', 1)[1]}"
        transformed_image.save(os.path.join(new_dataset_path, new_file_name))
