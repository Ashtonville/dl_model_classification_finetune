import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import false
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import default_collate
from Milestone_2.custom_attention import TemperatureMultiheadAttention

# define attention head replacer
def replace_attention(model):
    for layer in model.encoder.layers:
        old = layer.self_attention

        new = TemperatureMultiheadAttention(
            embed_dim=old.embed_dim,
            num_heads=old.num_heads,
            dropout=old.dropout,
            bias=True,
        )
        new.load_state_dict(old.state_dict(), strict=False)
        layer.self_attention = new

# define train algorithm, modified milestone 1 algorithm
def train(batch_size=256, use_cutmix=false, epochs=10, continue_training=False, use_new_attention=False):
    data_dir = "./data"
    info_dir = "./train_info"
    log_dir = info_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_vit"+ ("_cutmix" if use_cutmix else "")
    num_classes = 30
    batch_size = batch_size
    num_epochs = epochs
    learning_rate = 0.001

    # create log folder
    os.mkdir(log_dir)

    # set device to cluster gpu partition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print version information about cluster gpu and partition
    print("cuda_available:", torch.cuda.is_available())
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # transform to make sure image are correct size and convert to tensor
    tensor_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5], )
    ])


    # load train and val datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "aug_train_v2"), transform=tensor_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=tensor_transform)

    # use different dataloader if CutMix is used
    if use_cutmix:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  persistent_workers=True, collate_fn=collate_fn)
        print("\n Augmented.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    # scratch training or continue from pre-existing checkpoint
    if continue_training:
        state_dict = torch.load("./train_info/20251213-224327_vit_att_1/model.pth", map_location=device)
        model = models.vit_b_16(weights=None)
    else:
        model = models.vit_b_16(weights="IMAGENET1K_V1")

    # enable weight computation for backpropagation
    for param in model.parameters():
       param.requires_grad = True

    # replace generic architecture with fitting classifier and replace attention heads
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    if use_new_attention:
        replace_attention(model)

    # load pre-saved weights into custom model architecture to continue training
    if continue_training:
        model.load_state_dict(state_dict)
    
    model = model.to(device)

    # set criterion and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


    # log infos
    with open(os.path.join(log_dir, "info.txt"), "w") as f:
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"device: {print(torch.cuda.get_device_name(0))}\n")
        f.write(f"train size: {len(train_dataset)}\n")
        f.write(f"val size: {len(val_dataset)}\n")
        f.write(f"model: {model.__class__.__name__}\n")
        f.write(f"use_cutmix: {use_cutmix}\n")

    f_l = open(os.path.join(log_dir, "train_loss.txt"), "a")
    f_a = open(os.path.join(log_dir, "train_accuracy.txt"), "a")
    f_h = open(os.path.join(log_dir, "loss_history.txt"), "a")

    patience = 5  # number of epochs to wait for improvement
    min_improvement = 0.1
    best_val_acc = 0.0
    epochs_no_improve = 0

    best_model_acc = 0.0 # used to always save best model

    # training
    print("\nStart training")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training", unit="batch")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # label image as class with the highest probability
            if labels.ndim == 2:
                # soft labels (CutMix/MixUp)
                true_labels = labels.argmax(dim=1)
            else:
                # hard labels
                true_labels = labels
            # print(true_labels)

            correct += (predicted == true_labels).sum().item()

            train_loss = running_loss / total
            train_acc = 100 * correct / total
            progress_bar.set_postfix({"Loss": f"{train_loss:.4f}", "Acc": f"{train_acc:.2f}%"})

            # log loss etc.
            f_l.write(f"{train_loss:.4f}\n")
            f_a.write(f"{train_acc:.2f}\n")
            f_h.write(f"{loss.item():.4f}\n")

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

        # overwrite best_val_acc if improvement is significant
        if val_acc > best_model_acc:
            best_model_acc = val_acc
            # always save best model
            torch.save(model.state_dict(), os.path.join(log_dir, f"model.pth"))

        # get improvement compared to the best accuracy
        improvement = val_acc - best_val_acc

        # overwrite best_val_acc if there is a significant improvement
        if improvement > min_improvement:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # log loss etc.
        with open(os.path.join(log_dir, "epoch_val_loss.txt"), "a") as v_l:
            v_l.write(f"{val_loss:.4f}\n")

        with open(os.path.join(log_dir, "epoch_val_accuracy.txt"), "a") as v_a:
            v_a.write(f"{val_acc:.2f}\n")

        with open(os.path.join(log_dir, "epoch_train_loss.txt"), "a") as e_l:
            e_l.write(f"{train_loss:.4f}\n")

        with open(os.path.join(log_dir, "epoch_train_accuracy.txt"), "a") as e_a:
            e_a.write(f"{train_acc:.2f}\n")

        # removed to train full 30 epochs
        # stop training if there has been no significant improvement for some epochs
        #if epochs_no_improve >= patience:
            #print(f"No significant improvement for {patience} epochs. Trained {epoch + 1} epochs.")
            #break



    f_a.close()
    f_l.close()
    f_h.close()

    print("\n Training complete.")

def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=30)
    mixup = v2.MixUp(num_classes=30)
    cutmix_or_mixup = v2.RandomApply([v2.RandomChoice([cutmix, mixup])], p=0.3)
    return cutmix_or_mixup(*default_collate(batch))

if __name__ == "__main__":
    runs = [
        {"batch_size": 32, "use_cutmix":False},
    ]

    for cfg in runs:
        model_name = "ViT"
        print(f"\nStarting training: batch_size={cfg['batch_size']}, model={model_name}, cutmix={cfg['use_cutmix']}")

        try:
            train(**cfg)
            print(f"\nTraining completed for batch_size={cfg['batch_size']}, model={model_name}, cutmix={cfg['use_cutmix']}")
        except Exception as e:
            print(f"\nTraining FAILED for batch_size={cfg['batch_size']}, model={model_name}, cutmix={cfg['use_cutmix']}: {e}\n")
            continue  # move on to next run