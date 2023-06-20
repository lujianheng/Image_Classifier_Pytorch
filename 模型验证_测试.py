import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet34,resnet18,resnet101
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize([0.3280035786787705, 0.229786047379836, 0.16604132105721886],
                         [0.26646884675584875, 0.18641936393396857, 0.13503227047435348])
])

def test(dataloader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred.data, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        return y_true, y_pred

if __name__=='__main__':
    batch_size = 1

    full_data = datasets.ImageFolder("new_dataset")
    full_data.transform = test_transforms

    test_dataloader = DataLoader(dataset=full_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = resnet50(num_classes=5)
    model.load_state_dict(torch.load("log_data_resnet50_meanstd/resnet50_best.pth",map_location=device))
    model.to(device)

    y_true, y_pred = test(test_dataloader, model, device)

    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
