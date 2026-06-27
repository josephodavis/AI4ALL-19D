import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class BlindnessDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.ids[idx] + ".png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label