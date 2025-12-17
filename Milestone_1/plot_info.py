import os
import matplotlib.pyplot as plt

def plot_info(log_dir):
    info = {}
    with open(os.path.join(log_dir, 'info.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()

    model = info.get('model')
    batch_size = info.get('batch_size')
    learning_rate = info.get('learning_rate')
    cutmix=False
    if('cut mix' in info or "cutmix" in log_dir):
        cutmix=True


    with open(os.path.join(log_dir, "train_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.grid(True)
        plt.xlabel("Batch iteration")
        plt.ylabel("Loss")
        plt.suptitle(f"{model} Epoch Training Loss{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "train_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "loss_history.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.grid(True)
        plt.xlabel("Batch iteration")
        plt.ylabel("Loss")
        plt.suptitle(f"{model} Training Loss{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "loss_history.png"))
        plt.show()

    with open(os.path.join(log_dir, "train_accuracy.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.grid(True)
        plt.xlabel("Batch iteration")
        plt.ylabel("Accuracy")
        plt.suptitle(f"{model}  Training Accuracy{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0,100)
        plt.savefig(os.path.join(log_dir, "train_accuracy.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_train_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.suptitle(f"{model} Epoch Training Loss{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "epoch_train_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_train_accuracy.txt"), "r") as f:
        accuracies = [float(line.strip()) for line in f]
        plt.plot(accuracies)
        plt.yticks(range(0, 101, 10))
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.suptitle(f"{model} Epoch Training Accuracy{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0,100)
        plt.savefig(os.path.join(log_dir, "epoch_train_accuracy.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_val_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.suptitle(f"{model} Epoch Validation Loss{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "epoch_val_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_val_accuracy.txt"), "r") as f:
        accuracies = [float(line.strip()) for line in f]
        plt.plot(accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.suptitle(f"{model} Epoch Validation Accuracy{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0,100)
        plt.yticks(range(0, 101, 10))
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "epoch_val_accuracy.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_val_accuracy.txt"), "r") as f:
        accuracies = [float(line.strip()) for line in f]
        plt.plot(accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.suptitle(f"{model} Epoch Validation Accuracy{' using CutMix' if cutmix else ''}", fontsize=14)
        plt.title(f"Batch size: {batch_size}, Learning rate: {learning_rate}", fontsize=10)
        plt.ylim(0,70)
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'epoch_val_accuracy_close_{model}_{batch_size}_{"_cutix" if cutmix else ""}.png'))
        plt.show()

if __name__ == "__main__":
    plot_info(r"C:\Users\nikla\Desktop\dl_model_classification_finetune\ml2_results\vit_old")
