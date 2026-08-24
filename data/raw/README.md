Put the selected Kaggle CSV in this folder.

Example:
data/raw/landslide.csv

Do not commit the downloaded dataset to GitHub unless its Kaggle license allows redistribution.

The trainer can inspect a new CSV and infer numeric predictors automatically. For best results, pass the target explicitly:

python scripts/train_models.py --data data/raw/landslide.csv --target Landslide
