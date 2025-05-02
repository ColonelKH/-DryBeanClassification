import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv("data/preprocessed_scaled.csv")
X = df.drop(columns=['Class'])
y = df['Class']

# PCA
pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
avg_variance = explained_variance.mean()
n_components = sum(explained_variance > avg_variance)
X_pca_selected = X_pca[:, :n_components]

# Save PCA data
pca_df = pd.DataFrame(X_pca_selected)
pca_df['Class'] = y.values
pca_df.to_csv("data/pca_transformed.csv", index=False)

# Visualize first 2 PCA components
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title("PCA: First 2 Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.savefig("results/PCA_First_2_Components.png")
plt.close()

# LDA
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X, y)
lda_df = pd.DataFrame(X_lda)
lda_df['Class'] = y.values
lda_df.to_csv("data/lda_transformed.csv", index=False)

# Visualize first 2 LDA components
plt.figure(figsize=(8,6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title("LDA: First 2 Components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.savefig("results/LDA_First_2_Components.png")
plt.close()

print("Feature extraction complete. Files saved to data/ and results/")
