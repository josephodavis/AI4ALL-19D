"""Grad-CAM explainability for the blindness-detection CNN.

Grad-CAM (Selvaraju et al., 2017) highlights the image regions that most
influenced the model's score for a given class: the gradients of that class
score flowing into the last convolutional block weight its activation maps,
producing a coarse localization heatmap that is upsampled onto the input image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import IMG_SIZE, DeepCNN, preprocess_image


def compute_gradcam(
    model: DeepCNN,
    image: Image.Image,
    class_idx: int,
    device: str = "cpu",
) -> np.ndarray:
    """Return a Grad-CAM heatmap for ``class_idx`` as an (IMG_SIZE, IMG_SIZE) float array in [0, 1].

    Hooks the output of the conv stack (``features``) — the deepest spatial
    feature map before global average pooling. ``image`` is the raw upload; it
    goes through the same Ben Graham preprocessing as ``predict``.
    """
    activations: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    handle = model.features.register_forward_hook(forward_hook)
    try:
        tensor = preprocess_image(image, device=device)
        with torch.enable_grad():
            logits = model(tensor)
            # Differentiate only w.r.t. the captured activation so no parameter
            # gradients accumulate on the (cached, shared) model.
            grads = torch.autograd.grad(logits[0, class_idx], activations[0])[0]
    finally:
        handle.remove()

    acts = activations[0].detach()  # (1, C, H/32, W/32)
    grads = grads.detach()  # same shape as acts

    # Weight each channel by its average gradient, sum, and keep positive evidence.
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

    cam = cam[0, 0].cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map floats in [0, 1] to RGB uint8 colors (blue -> cyan -> yellow -> red)."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend the Grad-CAM heatmap onto an image.

    Pass the *model input* image (``model.model_input_image``), not the raw
    upload: Ben Graham preprocessing crops and re-squares the fundus, so a
    heatmap laid over the original would be spatially misaligned.

    ``alpha`` controls heatmap opacity; low-evidence regions stay mostly
    transparent so the fundus remains visible everywhere.
    """
    base = image.convert("RGB").resize(cam.shape[::-1])
    heat = _jet_colormap(cam).astype(np.float32)

    base_arr = np.asarray(base, dtype=np.float32)
    # Scale opacity by evidence strength so cold regions don't get tinted blue.
    weight = (alpha * cam[..., None]).astype(np.float32)
    blended = base_arr * (1.0 - weight) + heat * weight
    return Image.fromarray(blended.astype(np.uint8))
