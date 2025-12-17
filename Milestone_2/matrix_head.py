import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch_directml
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

# MLP-heads to reconstruct model

def head_relu(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

def head_gelu(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

def head_silu(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

def head_gelu_2_hidden(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

def head_bottleneck(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

def test_model(model_dir, batch_size, data_dir, title, head = None):
    num_classes = 30
    device = torch_directml.device(1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # setup model
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features

    if head is None:
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        model.heads = head(in_features, num_classes)


    # load weights
    state_dict = torch.load(os.path.join(model_dir, "model.pth"))
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    # run model
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Classifying"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)

    np.savetxt(
        os.path.join(model_dir, "confusion_matrix.txt"),
        cm,
        fmt="%d"
    )

    # confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes,
        square=True,
        annot_kws={"size": 8}
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Accuracy: {100 * correct / total:.2f}%")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix_counts.png"))
    plt.close()