"""Model definition, loading, and inference for the blindness-detection app.

The architecture mirrors ``DeepCNN`` in ``train_eval.ipynb`` (5 double-conv
blocks -> adaptive average pool -> 3-layer classifier head), so the checkpoint's
``state_dict`` loads with ``strict=True``.

Inference reproduces the *training* input pipeline exactly: images were trained
on the Ben Graham preprocessed corpus (``data/raw/2019_2015_data/resized_ben_graham``,
written offline at 512px by ``preprocess_ben_graham.py``) and then resized /
center-cropped to ``IMG_SIZE``. An uploaded image is therefore Ben Graham
preprocessed on the fly before it reaches the network — skipping that step
feeds the model a distribution it has never seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ``preprocess_ben_graham`` lives at the repo root, one level up from this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocess_ben_graham import ben_graham  # noqa: E402  (needs the sys.path line above)

# APTOS 2019 diabetic-retinopathy grades, in label order (index == diagnosis code).
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

CLASS_DESCRIPTIONS = {
    "No DR": "No diabetic retinopathy detected.",
    "Mild": "Mild non-proliferative diabetic retinopathy.",
    "Moderate": "Moderate non-proliferative diabetic retinopathy.",
    "Severe": "Severe non-proliferative diabetic retinopathy.",
    "Proliferative": "Proliferative diabetic retinopathy (most advanced stage).",
}

# Must match the notebook: IMG_SIZE is the network's input resolution, and
# BEN_GRAHAM_SIZE is the size the offline preprocessing wrote its images at
# (preprocess_ben_graham.py --img-size, default 512).
IMG_SIZE = 320
BEN_GRAHAM_SIZE = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split into geometry and tensor stages so the app can display the exact pixels
# the model sees (and Grad-CAM can overlay onto them) without re-deriving the
# resize/crop by hand.
_GEOMETRY = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
    ]
)

_TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def ben_graham_image(image: Image.Image) -> Image.Image:
    """Apply the offline Ben Graham pipeline to a PIL image.

    Crops to the fundus, pads to a square, resizes to ``BEN_GRAHAM_SIZE``, then
    high-pass filters and re-masks it — the same function that generated the
    training corpus, so uploads match what the model was fitted on.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(ben_graham(arr, img_size=BEN_GRAHAM_SIZE))


def model_input_image(image: Image.Image) -> Image.Image:
    """The exact IMG_SIZE x IMG_SIZE image the network sees, before normalization."""
    return _GEOMETRY(ben_graham_image(image))


def preprocess_image(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Ben Graham + resize/crop + normalize -> a (1, 3, IMG_SIZE, IMG_SIZE) batch."""
    return _TO_TENSOR(model_input_image(image)).unsqueeze(0).to(device)


def make_block(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    ]
    if dropout > 0:
        # Dropout2d zeroes entire channels rather than individual activations.
        # That is the right granularity for conv feature maps: neighbouring
        # pixels in a map are highly correlated, so plain Dropout leaves enough
        # of each feature intact that it barely regularizes.
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


DEEPCNN_WIDTH = "narrow"   # "narrow" | "wide"
DEEPCNN_WIDTHS = {"narrow": (32, 64, 128, 256, 512),
                  "wide":   (64, 128, 256, 512, 1024)}[DEEPCNN_WIDTH]
DEEPCNN_BATCH = {"narrow": 32, "wide": 16}[DEEPCNN_WIDTH]
CONV_DROPOUT = 0.15

class DeepCNN(nn.Module):
    def __init__(self, num_classes=5, conv_dropout=None, widths=None):
        super().__init__()
        conv_dropout = CONV_DROPOUT if conv_dropout is None else conv_dropout
        w = DEEPCNN_WIDTHS if widths is None else widths

        # Dropout only in the last two blocks. The early blocks learn generic
        # edge/color filters that aren't where memorization happens, and dropping
        # channels there mostly just slows convergence.
        self.features = nn.Sequential(
            make_block(3, w[0]),
            make_block(w[0], w[1]),
            make_block(w[1], w[2]),
            make_block(w[2], w[3], dropout=conv_dropout),
            make_block(w[3], w[4], dropout=conv_dropout),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(w[4], w[4] // 2),
            nn.BatchNorm1d(w[4] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(w[4] // 2, w[4] // 8),
            nn.ReLU(inplace=True),
            nn.Linear(w[4] // 8, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def load_model(model_path: str, device: str = "cpu") -> DeepCNN:
    """Load ``best_model.pth`` (a state_dict) into the model and set eval mode."""
    checkpoint = torch.load(model_path, map_location=device)
    # Support both raw state_dicts and {"state_dict": ...} style checkpoints.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model = DeepCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(model: DeepCNN, image: Image.Image, device: str = "cpu"):
    """Run inference on a PIL image.

    Returns a list of (class_name, probability) tuples in label order.
    """
    tensor = preprocess_image(image, device=device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return list(zip(CLASS_NAMES, probs))
