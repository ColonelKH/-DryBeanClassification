import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

# Load transformed datasets
raw = pd.read_csv("data/preprocessed_scaled.csv")
pca = pd.read_csv("data/pca_transformed.csv")
lda = pd.read_csv("data/lda_transformed.csv")

X_sets = {'Raw': raw.drop(columns='Class'), 'PCA': pca.drop(columns='Class'), 'LDA': lda.drop(columns='Class')}
y = raw['Class']

# Define models and params
models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [5, None]}),
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100]}),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {'n_estimators': [50, 100]}),
    'NaiveBayes': (GaussianNB(), {})
}

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

for rep_name, X in X_sets.items():
    for model_name, (model, param_grid) in models.items():
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for train_idx, test_idx in outer.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if param_grid:
                clf = GridSearchCV(model, param_grid, cv=inner, scoring='f1_macro')
                clf.fit(X_train, y_train)
                best_model = clf.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            scores['f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        results.append({
            'Model': model_name,
            'Representation': rep_name,
            'accuracy_mean': np.mean(scores['accuracy']),
            'accuracy_std': np.std(scores['accuracy']),
            'precision_mean': np.mean(scores['precision']),
            'recall_mean': np.mean(scores['recall']),
            'f1_mean': np.mean(scores['f1'])
        })

# Save results to CSV
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("results/final_metrics.csv", index=False)
print("Evaluation complete. Metrics saved to results/final_metrics.csv")

# ROC Curve: Logistic Regression on LDA
X_roc = X_sets['LDA']
y_bin = label_binarize(y, classes=np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X_roc, y_bin, test_size=0.2, stratify=y, random_state=42)
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i in range(y_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression on LDA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/ROC_Curve_LDA_LogReg.png")
plt.close()
print("ROC Curve saved to results/ROC_Curve_LDA_LogReg.png")
