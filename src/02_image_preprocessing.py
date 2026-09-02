import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR   = os.path.join(ROOT, "dataset", "image_dataset")
GRAPH_DIR = os.path.join(ROOT, "outputs", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)



# 1.  DATASET CLASS


class CampusImageDataset(Dataset):
    """Custom Dataset for campus location images."""

    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []       # list of (path, class_idx)
        self.classes   = []       # sorted class names
        self.class_to_idx = {}

        # Walk the directory structure
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.classes = class_dirs
        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}

        for cls in class_dirs:
            cls_path = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((
                        os.path.join(cls_path, fname),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



# 2.  TRANSFORM DEFINITIONS


# --- Basic (no augmentation) — used for validation / CLIP input ---
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet means
        std=[0.229, 0.224, 0.225]      # ImageNet stds
    ),
])

# --- Training transform (with augmentation) ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.3, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])



# 3.  CREATE DATASETS & DATALOADERS


print("\n" + "="*70)
print("  IMAGE PREPROCESSING PIPELINE")
print("="*70)

# Basic dataset (for inspection)
basic_dataset = CampusImageDataset(IMG_DIR, transform=basic_transform)
# Augmented dataset (for training)
train_dataset = CampusImageDataset(IMG_DIR, transform=train_transform)

print(f"\n  Total images   : {len(basic_dataset)}")
print(f"  Num classes    : {len(basic_dataset.classes)}")
print(f"  Classes        : {basic_dataset.classes}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(basic_dataset, batch_size=32, shuffle=False, num_workers=0)

# Grab a batch and print shape
batch_images, batch_labels = next(iter(train_loader))
print(f"\n  Batch shape    : {batch_images.shape}")
print(f"  Batch dtype    : {batch_images.dtype}")
print(f"  Labels shape   : {batch_labels.shape}")
print(f"  Pixel range    : [{batch_images.min():.3f}, {batch_images.max():.3f}]")



# 4.  VISUALISATION — BEFORE / AFTER AUGMENTATION


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverse ImageNet normalisation for display."""
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return torch.clamp(t, 0, 1)


print("\n  Generating augmentation samples...")

# Pick 4 different classes for demonstration
demo_classes = basic_dataset.classes[:4]
demo_indices = []
for cls in demo_classes:
    cls_idx = basic_dataset.class_to_idx[cls]
    for i, (_, label) in enumerate(basic_dataset.samples):
        if label == cls_idx:
            demo_indices.append(i)
            break

fig, axes = plt.subplots(len(demo_indices), 5, figsize=(18, 4*len(demo_indices)))
fig.suptitle("Image Preprocessing — Original vs Augmented Samples",
             fontsize=14, fontweight="bold", y=1.02)

for row, idx in enumerate(demo_indices):
    img_path, label = basic_dataset.samples[idx]
    cls_name = basic_dataset.classes[label]

    # Original (no transform)
    orig_img = Image.open(img_path).convert("RGB")
    orig_resized = orig_img.resize((224, 224))

    axes[row][0].imshow(orig_resized)
    axes[row][0].set_title(f"Original\n({cls_name})", fontsize=9)
    axes[row][0].axis("off")

    # 4 augmented versions
    for col in range(1, 5):
        aug_tensor = train_transform(orig_img)
        aug_img    = denormalize(aug_tensor).permute(1, 2, 0).numpy()
        axes[row][col].imshow(aug_img)
        axes[row][col].set_title(f"Augmented #{col}", fontsize=9)
        axes[row][col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "augmentation_samples.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print("  [✓] Saved: augmentation_samples.png")



# 5.  TRANSFORM PIPELINE SUMMARY


print(f"""
  ── Preprocessing Pipeline Summary ──

  VALIDATION / INFERENCE:
    1. Resize → 224 × 224
    2. ToTensor (0–1 float)
    3. Normalize (ImageNet mean/std)

  TRAINING (with augmentation):
    1. RandomResizedCrop(224, scale=0.7–1.0)
    2. RandomHorizontalFlip(p=0.5)
    3. RandomRotation(±15°)
    4. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    5. RandomGrayscale(p=0.05)
    6. ToTensor
    7. Normalize (ImageNet mean/std)

  DataLoader settings:
    Batch size      : 32
    Shuffle (train) : True
    Num workers     : 0 (Windows compatible)

[✓] Image preprocessing pipeline complete.
""")
