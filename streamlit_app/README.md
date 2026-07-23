# Blindness Level Detection — Streamlit App

A small web app that loads the trained CNN (`best_model.pth`) and predicts the
diabetic-retinopathy / blindness grade from an uploaded retinal **fundus photograph**.

Grades (APTOS 2019): **No DR · Mild · Moderate · Severe · Proliferative**

## Run

From the repo root, using the existing virtual environment:

```bash
pip install -r streamlit_app/requirements.txt   # streamlit is not in the repo's venv yet
streamlit run streamlit_app/app.py
```

Then open the URL Streamlit prints (default: http://localhost:8501).

## Usage

1. Upload a fundus image (`.png`, `.jpg`, `.jpeg`).
2. The app shows the predicted grade, confidence, and the full class-probability breakdown.
3. A **Grad-CAM** section (`explain.py`) overlays a heatmap on the image showing which
   retinal regions most influenced the selected grade's score. Pick any grade to explain
   and adjust the heatmap opacity; a map concentrated outside the retina is a hint the
   prediction may not be trustworthy for that image.

The model checkpoint is loaded from `../best_model.pth` (set as `MODEL_PATH` in `app.py`).

## Notes

- The model architecture in `model.py` is reconstructed to match the layers stored in
  `best_model.pth` (5 conv+BatchNorm blocks → adaptive average pool → BatchNorm'd FC head).
- Preprocessing mirrors the notebook's validation transform: resize to 224×224, `ToTensor`,
  and ImageNet normalization.
- **For educational/research use only — not a medical device.**
