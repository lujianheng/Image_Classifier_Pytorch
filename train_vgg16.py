import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置数据集路径
data_dir = "./new_dataset"
import os
os.environ['TORCH_HOME']='pretrain_model'

# 数据预处理
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    avg_loss = 0
    correct = 0
    model.train()
    progress_bar = tqdm(dataloader, desc='Training')
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': correct / ((batch + 1) * dataloader.batch_size)})
    avg_loss /= size
    avg_acc = correct / size
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', avg_acc, epoch)
    return avg_loss, avg_acc

def validate(dataloader, model, loss_fn, device, writer, epoch):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss /= size
    correct /= size
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} , F1:{f1:>8f}\n")
    writer.add_scalar('Validate/Loss', test_loss, epoch)
    writer.add_scalar('Validate/Accuracy', correct, epoch)
    writer.add_scalar('Validate/F1', f1, epoch)
    return correct, test_loss, f1

def WriteData(fname, *args):
    with open(fname, 'a+') as f:
        for data in args:
            f.write(str(data) + "\t")
        f.write("\n")

if __name__ == '__main__':
    batch_size = 128

    # 创建数据集
    full_data = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(full_data))  # 80%的数据用于训练
    test_size = len(full_data) - train_size
    train_data, valid_data = random_split(full_data, [train_size, test_size])

    # Apply separate transforms to the training and validation data
    train_data.dataset.transform = train_transforms
    valid_data.dataset.transform = test_transforms

    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    vgg16 = models.vgg16(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = False
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, 5)
    
    
    nn.init.xavier_normal_(vgg16.classifier[6].weight)
    vgg16 = vgg16.to(device)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(vgg16.parameters(), lr=learning_rate, weight_decay=1e-4)

    epochs = 100
    loss_ = 10
    save_root = "./log_data_vgg16/"
    writer = SummaryWriter('./runs/experiment_vgg16/')

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        time_start = time.time()
        avg_loss, avg_acc = train(train_dataloader, vgg16, loss_fn, optimizer, device, writer, t)
        time_end = time.time()
        print(f"train time: {(time_end - time_start)}")
        val_accuracy, val_loss, val_f1 = validate(valid_dataloader, vgg16, loss_fn, device, writer, t)
        WriteData(save_root + "vgg16.txt",
                  "epoch", t,
                  "train_loss", avg_loss,
                  "train_accuracy", avg_acc,
                  "val_loss", val_loss,
                  "val_accuracy", val_accuracy,
                  "val_f1", val_f1)
        if t % 10 == 0:
            torch.save(vgg16.state_dict(), save_root + "vgg16_epoch" + str(t) + "_loss_" + str(avg_loss) + ".pth")
        torch.save(vgg16.state_dict(), save_root + "vgg16_last.pth")
        if avg_loss < loss_:
            loss_ = avg_loss
            torch.save(vgg16.state_dict(), save_root + "vgg16_best.pth")

    writer.close()
