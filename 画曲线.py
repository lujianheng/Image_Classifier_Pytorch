import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志文件
with open("log_data_resnet101_meanstd/resnet101.txt", "r") as f:
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

# 创建一个新的figure
plt.figure(figsize=(12, 8))

# 创建第一个子图，显示train_loss和val_loss
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='train_loss')
plt.plot(df['epoch'], df['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 创建第二个子图，显示train_accuracy和val_accuracy
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['train_accuracy'], label='train_accuracy')
plt.plot(df['epoch'], df['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 创建第三个子图，显示val_f1
plt.subplot(2, 1, 2)
plt.plot(df['epoch'], df['val_f1'], label='val_f1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
