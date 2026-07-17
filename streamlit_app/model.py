"""Model definition, loading, and inference for the blindness-detection app.

The architecture here is reconstructed from the layers stored in ``best_model.pth``
(5 conv+BatchNorm blocks -> adaptive average pool -> BatchNorm'd FC head), so the
checkpoint's ``state_dict`` loads with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# APTOS 2019 diabetic-retinopathy grades, in label order (index == diagnosis code).
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

CLASS_DESCRIPTIONS = {
    "No DR": "No diabetic retinopathy detected.",
    "Mild": "Mild non-proliferative diabetic retinopathy.",
    "Moderate": "Moderate non-proliferative diabetic retinopathy.",
    "Severe": "Severe non-proliferative diabetic retinopathy.",
    "Proliferative": "Proliferative diabetic retinopathy (most advanced stage).",
}

# Matches the notebook's deterministic validation transform.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class BlindnessCNN(nn.Module):
    """From-scratch CNN for 5-class diabetic-retinopathy grading."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path: str, device: str = "cpu") -> BlindnessCNN:
    """Load ``best_model.pth`` (a state_dict) into the model and set eval mode."""
    checkpoint = torch.load(model_path, map_location=device)
    # Support both raw state_dicts and {"state_dict": ...} style checkpoints.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model = BlindnessCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(model: BlindnessCNN, image: Image.Image, device: str = "cpu"):
    """Run inference on a PIL image.

    Returns a list of (class_name, probability) tuples in label order.
    """
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return list(zip(CLASS_NAMES, probs))
