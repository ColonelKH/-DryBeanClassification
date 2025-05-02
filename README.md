# Dry Bean Classification - Machine Learning Project

This repository contains a complete machine learning pipeline for classifying dry bean types using the Dry Bean Dataset.

## 📁 Project Structure

```
DryBeanClassification/
├── 01_preprocessing.py            # Preprocessing: missing values, outliers, scaling
├── 02_feature_extraction.py       # Feature extraction using PCA and LDA
├── 03_modeling_evaluation.py      # Modeling, nested cross-validation, ROC curves
├── requirements.txt               # Required Python packages
├── README.md                      # Project documentation
├── data/                          # Raw and processed datasets
├── results/                       # Final metrics and visual outputs
└── reports/                       # Final report document
```

## 📊 Dataset

- **Source**: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- 13,611 samples, 7 classes, 16 numerical features

## ⚙️ Setup

1. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the original dataset file into the `data/` directory as `Dry_Bean_Dataset.xlsx`.

4. Run scripts in order:
    ```bash
    python 01_preprocessing.py
    python 02_feature_extraction.py
    python 03_modeling_evaluation.py
    ```

## 📈 Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- XGBoost

## ✅ Features

- Missing value simulation and handling
- Z-score-based outlier clipping
- Standard scaling
- Dimensionality reduction via PCA and LDA
- Nested cross-validation (5x3 folds)
- ROC Curve analysis (One-vs-Rest)

## 🏁 Results

- Best performing combo: **Logistic Regression + LDA**
- Achieved high F1 and ROC-AUC scores for multiple classes

## 👨‍💻 Author

- Hadi Nafeh
