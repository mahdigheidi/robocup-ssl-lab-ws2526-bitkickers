import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt


def plot_rewards(rewards, smooth_window=20):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.4)
    avg = np.convolve(rewards, np.ones(smooth_window) / smooth_window, mode="valid")
    plt.plot(range(19, len(rewards)), avg, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TD3 + HER Training")
    plt.grid()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"plot_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"plot_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()
