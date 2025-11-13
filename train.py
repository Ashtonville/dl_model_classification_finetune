import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import torch_directml
from tqdm import tqdm



def train():
    data_dir = "./data"
    info_dir = "./train_info"
    log_dir = info_dir+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    num_classes = 30
    batch_size = 256
    num_epochs = 2
    learning_rate = 1e-3

    # create log folder
    os.mkdir(log_dir)

    # direct ml to use AMD GPU
    # index to use actual GPU and not internal
    device = torch_directml.device(1)

    # transform to make sure image are correct size and convert to tensor
    tensor_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # load dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "aug_train"), transform=tensor_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=tensor_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    # configure model
    model = models.resnet18(weights="IMAGENET1K_V1")

    # change output to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # log infos
    with open(os.path.join(log_dir, "info.txt"), "w") as f:
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"device: {torch_directml.device_name(1)}\n")
        f.write(f"train size: {len(train_dataset)}\n")
        f.write(f"val size: {len(val_dataset)}\n")

    loss_history = []

    # training
    print("\nStart training")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training", unit="batch")
        with open(os.path.join(log_dir, "train_loss.txt"), "a") as f_l:
            with open(os.path.join(log_dir, "train_accuracy.txt"), "a") as f_a:
                for batch_idx, (images, labels) in enumerate(progress_bar):
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    train_loss = running_loss / total
                    train_acc = 100 * correct / total
                    progress_bar.set_postfix({"Loss": f"{train_loss:.4f}", "Acc": f"{train_acc:.2f}%"})

                    # log loss etc.
                    f_l.write(f"{train_loss:.4f}\n")
                    f_a.write(f"{train_acc:.2f}\n")

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation", unit="batch")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix(
                    {"Loss": f"{val_loss / val_total:.4f}", "Acc": f"{100 * val_correct / val_total:.2f}%"})

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / val_total
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}] done — "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s\n")

        # log loss etc.
        with open(os.path.join(log_dir, "epoch_val_loss.txt"), "a") as v_l:
            v_l.write(f"{val_loss:.4f}\n")

        with open(os.path.join(log_dir, "epoch_val_accuracy.txt"), "a") as v_a:
            v_a.write(f"{val_acc:.2f}\n")

        with open(os.path.join(log_dir, "epoch_train_loss.txt"), "a") as e_l:
            e_l.write(f"{train_loss:.4f}\n")

        with open(os.path.join(log_dir, "epoch_train_accuracy.txt"), "a") as e_a:
            e_a.write(f"{train_acc:.2f}\n")

    with open(os.path.join(log_dir, "loss_history.txt"), "w") as f:
        for loss in loss_history:
            f.write(f"{loss:.4f}\n")
    # save model
    print("\n Training complete.")
    torch.save(model.state_dict(), os.path.join(log_dir,f"resnet.pth"))


if __name__ == "__main__":

    train()
