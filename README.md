# AI4ALL-19D
AI4ALL Group 19D Project Repo

## Setup Instructions
1. After creating your virtual environment, in your terminal, run: <code>pip install -r requirements.txt</code>
2. Head to Kaggle: [Resized 2015-2019 Diabetic Retinopathy Detection](https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection)
3. From the Kaggle data card, make sure to download: 
   - Labels: traintestLabels15_trainLabels19.csv.zip (located under the labels folder)
   - Images: resized_traintest15_train19.zip (~18 GB compressed)
4. Inside the `data/raw/` folder, create a folder called `2019_2015_data` and put the downloaded image folder and csv into it.
5. To train, run <code>python model/train_eval.py</code>

## NOTE ON DATA PATHS (not to be part of official README)
- local paths vary OS to OS, the original path given by Joseph's code I tried using and for some reason my computer refused to do it even though there was no clear reason why so I had IDE Agent fixed it using some fix, this may not work for you so if it doesn't, change your paths to where your datasets are.
- Lines 90 to 100 is how I loaded data