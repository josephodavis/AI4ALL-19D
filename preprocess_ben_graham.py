#!/usr/bin/env python3
"""Ben Graham fundus preprocessing (offline, one-time).

Reproduces the preprocessing from the 2015 Kaggle Diabetic Retinopathy winning
solution, which makes lesions (microaneurysms, hemorrhages, exudates) far more
visible and normalizes color/exposure differences between the 2015 and 2019
sub-datasets. Per image:

  1. Crop to the fundus by a grayscale threshold (removes the black border).
  2. Pad to a square so the circular fundus is not distorted.
  3. Resize to a fixed square size.
  4. Local-average-color subtraction:  out = 4*(img - gaussian_blur(img)) + 128
     This is a high-pass filter: flat regions go grey (~128), fine structure pops.
  5. Re-apply a circular mask, filling the corners with 128, to kill the bright
     ring artifact the blur-subtraction leaves at the crop boundary.

Runs the whole corpus in parallel and is resumable (skips images already done).
It writes to a NEW sibling directory so the original images stay untouched for
A/B comparison. Point IMAGE_DIR in train_eval.ipynb at the output directory to
train on the preprocessed images.

Examples
--------
Preview 12 before/after pairs (tune params before the full run)::

    python preprocess_ben_graham.py --sample 12

Process the whole training corpus::

    python preprocess_ben_graham.py

Custom paths / resolution::

    python preprocess_ben_graham.py --input-dir path/to/in --output-dir path/to/out --img-size 512
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Default paths mirror train_eval.ipynb's layout.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "2019_2015_data"
DEFAULT_INPUT_DIR = DATA_ROOT / "resized_traintest15_train19"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "resized_ben_graham"

IMAGE_EXTS = (".png", ".jpeg", ".jpg")


def crop_to_fundus(img, tol=7):
    """Crop away the near-black border by a grayscale threshold.

    img: HxWx3 uint8 array. Returns the tight bounding box around pixels whose
    mean intensity exceeds ``tol``. If the image is essentially all black, the
    original is returned unchanged.
    """
    gray = img.mean(axis=2)
    mask = gray > tol
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    """Zero-pad the shorter side so the fundus circle keeps its aspect ratio."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant")


def circular_mask(size, radius_frac=0.9):
    """Boolean HxW mask, True inside a centered circle of radius_frac * (size/2)."""
    center = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    r = center * radius_frac
    return (xx - center) ** 2 + (yy - center) ** 2 <= r ** 2


def ben_graham(img, img_size=512, sigma=None, radius_frac=0.9):
    """Full pipeline on a single RGB uint8 array -> processed uint8 array."""
    img = crop_to_fundus(img)
    img = pad_to_square(img)

    # Resize to a fixed square with PIL (high-quality Lanczos).
    img = np.asarray(
        Image.fromarray(img).resize((img_size, img_size), Image.LANCZOS),
        dtype=np.float32,
    )

    # Local-average-color subtraction. Canonical Ben Graham uses sigma = radius/30;
    # here radius ~ img_size/2, so img_size/30 is a good default. Blur spatially
    # only (sigma 0 on the channel axis) so colours don't bleed together.
    if sigma is None:
        sigma = img_size / 30.0
    blurred = gaussian_filter(img, sigma=(sigma, sigma, 0))
    out = 4.0 * (img - blurred) + 128.0
    out = np.clip(out, 0, 255)

    # Trim the outer ring where the subtraction leaves an artifact; fill with 128.
    mask = circular_mask(img_size, radius_frac)[:, :, None]
    out = out * mask + 128.0 * (~mask)
    return out.astype(np.uint8)


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def process_one(name, input_dir, output_dir, img_size, sigma, radius_frac, overwrite):
    """Worker: process a single filename. Returns (status, name[, error])."""
    out_path = output_dir / name
    if out_path.exists() and not overwrite:
        return ("skipped", name)
    try:
        img = _load_rgb(input_dir / name)
        processed = ben_graham(img, img_size=img_size, sigma=sigma, radius_frac=radius_frac)
        out_img = Image.fromarray(processed)
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            out_img.save(out_path, quality=95, subsampling=0)
        else:
            out_img.save(out_path)
        return ("done", name)
    except Exception as exc:  # keep the run going; report at the end
        return ("error", name, f"{type(exc).__name__}: {exc}")


def list_images(input_dir):
    return sorted(f.name for f in input_dir.iterdir()
                  if f.suffix.lower() in IMAGE_EXTS and f.is_file())


def save_preview(names, input_dir, img_size, sigma, radius_frac, out_path):
    """Save a before/after grid for eyeballing parameters."""
    import matplotlib.pyplot as plt

    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, name in enumerate(names):
        img = _load_rgb(input_dir / name)
        before = np.asarray(
            Image.fromarray(pad_to_square(crop_to_fundus(img))).resize(
                (img_size, img_size), Image.LANCZOS
            )
        )
        after = ben_graham(img, img_size=img_size, sigma=sigma, radius_frac=radius_frac)
        axes[row, 0].imshow(before)
        axes[row, 0].set_title(f"{name}\noriginal (cropped)", fontsize=8)
        axes[row, 1].imshow(after)
        axes[row, 1].set_title("Ben Graham", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview grid to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ben Graham fundus preprocessing.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-size", type=int, default=512,
                        help="output square size in px (>= training IMG_SIZE)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian blur sigma; default = img_size / 30")
    parser.add_argument("--radius-frac", type=float, default=0.9,
                        help="circular-mask radius as a fraction of img_size/2")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess images even if the output already exists")
    parser.add_argument("--sample", type=int, default=0,
                        help="preview N before/after pairs and exit (no output written)")
    parser.add_argument("--preview-path", type=Path,
                        default=PROJECT_ROOT / "ben_graham_preview.png")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N images (for quick tests)")
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        sys.exit(f"Input dir does not exist: {args.input_dir}")

    names = list_images(args.input_dir)
    if not names:
        sys.exit(f"No images ({', '.join(IMAGE_EXTS)}) found in {args.input_dir}")
    print(f"Found {len(names)} images in {args.input_dir}")

    # Preview mode: deterministic, evenly-spaced sample so both 2015 and 2019 show up.
    if args.sample > 0:
        k = min(args.sample, len(names))
        step = max(1, len(names) // k)
        picks = names[::step][:k]
        save_preview(picks, args.input_dir, args.img_size, args.sigma,
                     args.radius_frac, args.preview_path)
        return

    if args.limit > 0:
        names = names[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {args.output_dir} | img_size={args.img_size} "
          f"| sigma={args.sigma if args.sigma is not None else args.img_size / 30:.2f} "
          f"| workers={args.workers} | overwrite={args.overwrite}")

    worker = partial(
        process_one,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        sigma=args.sigma,
        radius_frac=args.radius_frac,
        overwrite=args.overwrite,
    )

    counts = {"done": 0, "skipped": 0, "error": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for result in tqdm(ex.map(worker, names), total=len(names), desc="Ben Graham"):
            counts[result[0]] += 1
            if result[0] == "error":
                errors.append(result[1:])

    print(f"\nDone: {counts['done']} processed, {counts['skipped']} skipped, "
          f"{counts['error']} errors.")
    if errors:
        print("First errors:")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")

    n_out = len(list_images(args.output_dir))
    print(f"Output directory now holds {n_out} images "
          f"(expected {len(list_images(args.input_dir))} for a complete run).")


if __name__ == "__main__":
    main()
