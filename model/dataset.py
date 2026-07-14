import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class BlindnessDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, id_col="image", label_col="level"):
        self.ids = df[id_col].values
        self.labels = df[label_col].values
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.image_dir, self.ids[idx] + ".png")
        # image = Image.open(img_path).convert("RGB")
        # image = self.transform(image)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)
        # return image, label

        img_name = str(self.ids[idx])
        
        # Dynamically discover the file extension (.jpeg, .png, or .jpg)
        if not img_name.endswith(('.png', '.jpeg', '.jpg')):
            img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_dir, f"{img_name}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        else:
            img_path = os.path.join(self.image_dir, img_name)

        # Open image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing image: {img_path}. Verify image_dir path.")
            
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label