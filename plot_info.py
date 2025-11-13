import os
import matplotlib.pyplot as plt

def plot_info(log_dir):
    with open(os.path.join(log_dir, "train_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.xlabel("Batch iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "train_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "loss_history.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.xlabel("Batch iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "loss_history.png"))
        plt.show()

    with open(os.path.join(log_dir, "train_accuracy.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.xlabel("Batch iteration")
        plt.ylabel("Accuracy")
        plt.title("Training Accurcacy")
        plt.savefig(os.path.join(log_dir, "train_accuracy.png"))
        plt.ylim(0,100)
        plt.show()

    with open(os.path.join(log_dir, "epoch_train_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.xticks(range(0, len(losses)))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch Training Loss")
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "epoch_train_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_train_accuracy.txt"), "r") as f:
        accuracies = [float(line.strip()) for line in f]
        plt.plot(accuracies)
        plt.xticks(range(0, len(losses)))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Epoch Training Accurcacy")
        plt.savefig(os.path.join(log_dir, "epoch_train_accuracy.png"))
        plt.ylim(0,100)
        plt.show()

    with open(os.path.join(log_dir, "epoch_val_loss.txt"), "r") as f:
        losses = [float(line.strip()) for line in f]
        plt.plot(losses)
        plt.xticks(range(0, len(losses)))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch Validation Loss")
        plt.ylim(0)
        plt.savefig(os.path.join(log_dir, "epoch_val_loss.png"))
        plt.show()

    with open(os.path.join(log_dir, "epoch_val_accuracy.txt"), "r") as f:
        accuracies = [float(line.strip()) for line in f]
        plt.plot(accuracies)
        plt.xticks(range(0, len(losses)))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Epoch Validation Accurcacy")
        plt.savefig(os.path.join(log_dir, "epoch_val_accuracy.png"))
        plt.ylim(0,100)
        plt.show()


plot_info("./train_info/20251113-142658")
