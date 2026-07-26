import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Define seed_worker here so Windows worker processes can import it natively
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class BlindnessDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, id_col="image", label_col="level"):
        self.ids = df[id_col].values
        self.labels = df[label_col].values
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()

        # Pre-index image paths once on initialization
        self.img_paths = []
        for img_id in self.ids:
            img_name = str(img_id)
            if not img_name.endswith((".png", ".jpeg", ".jpg")):
                img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f"{img_name}.png")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
            else:
                img_path = os.path.join(self.image_dir, img_name)
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing image: {img_path}. Verify image_dir path.")

        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label