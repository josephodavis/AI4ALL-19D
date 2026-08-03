# AI4ALL-19D

AI4ALL Group 19D Project Repo — https://blindnessdetection.streamlit.app/

Diabetic retinopathy severity grading (5 classes: No DR / Mild / Moderate / Severe /
Proliferative) on the combined 2015 + 2019 fundus image datasets.

## 1. Environment setup

After creating and activating a virtual environment, install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Download the data

1. Download the dataset from
   [Resized 2015-2019 Diabetic Retinopathy Detection](https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection).
   From the Kaggle data card you need two files:
    - **Labels:** `traintestLabels15_trainLabels19.csv.zip` (under the `labels` folder)
    - **Images:** `resized_traintest15_train19.zip` (~18 GB compressed)
2. Unzip both into `data/raw/` so the folder structure looks like this:
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

> **Note on paths.** The notebook resolves everything relative to the repo root
> (`project_root = Path.cwd()`), but local layouts still vary between machines. If the
> path checks fail, edit the **Paths** cell to point at wherever your data actually lives
> rather than moving the data around.

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

| Setting         | Options                            | Notes                                                                                                  |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `MODEL_TYPE`    | `"cnn"` / `"resnet"` / `"resnet50"` | `cnn` = from-scratch `DeepCNN` (25 epochs, lr 3e-4, `IMG_SIZE`); `resnet` = pretrained ResNet18 (10 epochs, lr 1e-4, 224); `resnet50` = pretrained ResNet50 with frozen backbone (10 epochs, lr 3e-4, 224) |
| `LOSS_TYPE`     | `"ce"` / `"sord"`                  | `ce` = cross-entropy; `sord` = soft ordinal loss (partial credit for near-miss grades), width `SORD_SIGMA` |
| `IMG_SIZE`      | e.g. `320`                         | input resolution for the CNN; both ResNet branches override this back to 224                            |
| `SAMPLER_POWER` | `0.0`–`1.0`                        | class-imbalance sampling strength. `1.0` = fully balanced batches, `0.5` = square-root middle ground (default), `0.0` = natural distribution |
| `NUM_WORKERS`   | e.g. `2`                           | DataLoader workers. Safe to raise above 0 because the Dataset/transforms live in the importable `dr_data.py`, so `spawn` workers can pickle them |
| `USE_AMP`       | `True` / `False`                   | mixed-precision training on CUDA (~2–3x faster, ~30% less VRAM). Ignored on CPU/MPS, which stay FP32   |

Training auto-selects the fastest available device (CUDA → Apple MPS → CPU).
Each run writes, keyed by `MODEL_TYPE`:

- `<model>_best_model.pth` — checkpoint from the **highest validation macro-F1** epoch
- `<model>_model.pth` — final-epoch weights
- `<model>_confusion_matrix.png`, `<model>_loss_curves.png` — evaluation plots

The final evaluation prints a per-class report, macro-F1, quadratic weighted kappa,
and a breakdown by data source (2015 vs 2019).

Checkpoints are gitignored apart from `best_model.pth`, which the Streamlit app loads.

## 5. Run the Streamlit app

The app loads `best_model.pth` and predicts a grade from an uploaded fundus photograph:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Alongside the prediction it renders a **Grad-CAM** heatmap showing which retinal regions
drove the score for the selected grade. See [`streamlit_app/README.md`](streamlit_app/README.md)
for details.

This tool is for educational and research purposes only. It is **not** a medical device
and must not be used for diagnosis.
