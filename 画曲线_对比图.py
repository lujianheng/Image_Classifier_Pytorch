import pandas as pd
import matplotlib.pyplot as plt

# 定义模型的日志文件路径
log_files = {
    "resnet50": "log_data_resnet50/resnet50.txt",
    "alexnet": "log_data_alexnet/alexnet.txt",
    "vgg16": "log_data_vgg16/vgg16.txt"
}

# 创建一个新的figure
plt.figure(figsize=(12, 8))

for model_name, log_file in log_files.items():
    # 读取训练日志文件
    with open(log_file, "r") as f:
        lines = f.readlines()

    # 处理每一行，将键值对转换为字典
    data = []
    for line in lines:
        items = line.split("\t")
        # 忽略最后一个空字符串
        data.append({items[i]: float(items[i + 1]) for i in range(0, len(items) - 1, 2)})

    # 将数据转换为DataFrame
    df = pd.DataFrame(data)

    # 将'epoch'列加1，使其从1开始
    df['epoch'] = df['epoch'] + 1

    # 创建第一个子图，显示train_loss和val_loss
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label=f'{model_name} train_loss')
    plt.plot(df['epoch'], df['val_loss'], label=f'{model_name} val_loss')

    # 创建第二个子图，显示train_accuracy和val_accuracy
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train_accuracy'], label=f'{model_name} train_accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label=f'{model_name} val_accuracy')

    # 创建第三个子图，显示val_f1
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['val_f1'], label=f'{model_name} val_f1')

# 设置子图1的标题和图例
plt.subplot(2, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 设置子图2的标题和图例
plt.subplot(2, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 设置子图3的标题和图例
plt.subplot(2, 1, 2)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
