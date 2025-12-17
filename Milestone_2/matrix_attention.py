import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

# set configuration for cluster environment
TEST_DIR = "./data/test"
MODEL_PATH = "./train_info/20251214-213106_vit_att_2/model.pth"
OUTPUT_DIR = "./eval_results"
NUM_CLASSES = 30
BATCH_SIZE = 32

# set device to cluster gpu partition
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create log folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# setup transforms for images
tensor_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5], )
    ])

# setup test dataset and loader
test_dataset = datasets.ImageFolder(TEST_DIR, transform=tensor_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    persistent_workers=True,
)

class_names = test_dataset.classes

print(f"Test samples: {len(test_dataset)}")
print("Class mapping:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# setup model with temperature scaling multiheadattention
from Milestone_2.custom_attention import TemperatureMultiheadAttention

# load pre-saved weights and generic vit model
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model = models.vit_b_16(weights=None)

# define replacement method
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


# replace generic architecture with fitting classifier and replace attention heads
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
replace_attention(model)

# load weights and set model to evaluation
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Model loaded")

# model evaluation
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Classifying"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # für Accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # für Confusion Matrix
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

accuracy = 100.0 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# setup matrix
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=np.arange(NUM_CLASSES)
)

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)

# plot matrix
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
plt.title(f"Accuracy: {accuracy:.2f}%")
plt.suptitle("ViT Scaled-Temperature Attention")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_counts.png"))
plt.close()

print(f"Confusion matrix image saved to: {OUTPUT_DIR}")

# Save additional summary
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Samples: {total}\n")