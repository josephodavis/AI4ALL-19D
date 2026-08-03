"""Data-pipeline pieces that DataLoader worker processes must be able to import.

Why this file exists: on Windows/macOS, DataLoader workers use the 'spawn' start
method, which re-imports the entry module in each worker. Objects defined in
notebook cells live in '__main__' and can't be found by spawned workers
(-> "Can't get attribute 'BlindnessDataset' on <module '__main__'>"), which is why
num_workers used to be pinned to 0. Defining them in this importable module lets
num_workers > 0 work, so image decode/augmentation runs in parallel and is hidden
behind GPU compute.

This cell WRITES the file; the next cell imports from it. Keep SEED in sync with the
notebook's SEED if you change it.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BlindnessDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, id_col="image", label_col="level"):
        self.ids = df[id_col].values
        self.labels = df[label_col].values
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = str(self.ids[idx])

        # Dynamically discover the file extension (.jpeg, .png, or .jpg)
        if not img_name.endswith((".png", ".jpeg", ".jpg")):
            img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_dir, f"{img_name}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        else:
            img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing image: {img_path}. Verify image_dir path.")

        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_transforms(img_size=320):
    # Fundus images have no canonical orientation, so flips and full rotation are
    # valid. hue is left untouched because color carries diagnostic signal
    # (hemorrhages, exudates). RandomResizedCrop outputs a square, which also fixes
    # the aspect-ratio squash from the old Resize((224, 224)) on non-square images.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Validation is deterministic: resize shortest side then center-crop to a
    # square (no squash, no augmentation).
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
