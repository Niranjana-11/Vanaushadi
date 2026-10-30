"""
Shows a grid of training images from two species side by side, so you can
eyeball whether your dataset explains a confusion pattern -- e.g. mislabeled
images, photos that include flowers/fruit instead of just leaves, or genuinely
similar-looking leaves.

Run this on the SAME machine where your `final_dataset` folder lives
(usually inside Colab, not your laptop, unless you downloaded it).

Usage:
  python inspect_confusion.py Tulsi Curry_Leaf
"""

import sys
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "final_dataset"
N_PER_CLASS = 6


def show_class_grid(class_a, class_b):
    fig, axes = plt.subplots(2, N_PER_CLASS, figsize=(3 * N_PER_CLASS, 6))

    for row, cls in enumerate([class_a, class_b]):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            print(f"Folder not found: {cls_path}")
            print("Available folders:", os.listdir(DATA_DIR))
            return
        files = os.listdir(cls_path)
        sample = random.sample(files, min(N_PER_CLASS, len(files)))
        for col in range(N_PER_CLASS):
            ax = axes[row][col]
            if col < len(sample):
                img = Image.open(os.path.join(cls_path, sample[col]))
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=14)
        axes[row][0].text(-0.1, 0.5, cls, fontsize=14, fontweight="bold",
                           rotation=90, va="center", ha="center",
                           transform=axes[row][0].transAxes)

    plt.suptitle(f"Training images: {class_a} (top) vs {class_b} (bottom)", fontsize=16)
    plt.tight_layout()
    plt.savefig("confusion_comparison.png", dpi=120, bbox_inches="tight")
    print("Saved comparison grid to confusion_comparison.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inspect_confusion.py <ClassA> <ClassB>")
        print("Example: python inspect_confusion.py Tulsi Curry_Leaf")
        sys.exit(1)
    show_class_grid(sys.argv[1], sys.argv[2])
