# AI4ALL-19D
AI4ALL Group 19D Project Repo - https://blindnessdetection.streamlit.app/

Diabetic retinopathy severity grading (5 classes: No DR / Mild / Moderate / Severe /
Proliferative) on the combined 2015 + 2019 fundus image datasets.

## 1. Environment setup
After creating and activating a virtual environment, install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Download the data
1. Download the dataset from
   https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection
2. Unzip it into `data/raw/` so the folder structure looks like this:
   ```
   data/raw/2019_2015_data/
   ├── labels/
   │   └── traintestLabels15_trainLabels19.csv   # image ids + severity labels
   ├── resized_traintest15_train19/              # training images (2015 + 2019)
   └── resized_test19/                           # 2019 test images
   ```
   The notebook reads the CSV from `labels/` and images from
   `resized_traintest15_train19/`. If the first cells report `CSV exists: False`,
   double-check this layout.

## 3. (Recommended) Ben Graham preprocessing
This applies fundus preprocessing (crop to the retina, subtract the local-average color, mask the border). It makes small lesions far more visible and normalizes color/exposure differences between the 2015 and 2019 images.

```bash
# Preview 12 before/after pairs first to sanity-check the output (writes ben_graham_preview.png)
python preprocess_ben_graham.py --sample 12

# Process the whole dataset -> data/raw/2019_2015_data/resized_ben_graham/
python preprocess_ben_graham.py
```
The full run takes roughly 80 minutes and uses ~9 GB of disk. It is parallelized
and **resumable** — re-running skips images that are already done. Useful flags:
`--img-size` (default 512), `--sigma`, `--workers`, `--limit N` (quick test),
`--overwrite`.

To train on the preprocessed images, set `IMAGE_DIR` in the paths cell of
`train_eval.ipynb` to `DATA_ROOT / "resized_ben_graham"`.

## 4. Train the model
Open `train_eval.ipynb` and run the cells top to bottom. The last cell starts
training. Configure the run in the **Configuration** cell at the top:

| Setting | Options | Notes |
|---------|---------|-------|
| `MODEL_TYPE` | `"cnn"` / `"resnet"` | `cnn` = from-scratch `FirstCNN`; `resnet` = ImageNet-pretrained ResNet18 baseline |
| `LOSS_TYPE` | `"ce"` / `"sord"` | `ce` = cross-entropy; `sord` = soft ordinal loss (partial credit for near-miss grades) |
| `IMG_SIZE` | e.g. `320` | input resolution for the CNN (ResNet stays at 224) |

Training auto-selects the fastest available device (CUDA → Apple MPS → CPU).
Each run writes, keyed by `MODEL_TYPE`:
- `<model>_best_model.pth` / `<model>_model.pth` — best-val-loss and final checkpoints
- `<model>_confusion_matrix.png`, `<model>_loss_curves.png` — evaluation plots

The final evaluation prints a per-class report, macro-F1, quadratic weighted kappa,
and a breakdown by data source (2015 vs 2019).