import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# Load the dataset
# df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")  # Uncomment when file is present

# Simulate a dataset structure (for development/testing without file)
df = pd.DataFrame(np.random.rand(1000, 16), columns=[
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
])
df['Class'] = np.random.choice(['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON'], 1000)

# Add missing values
for col in ['Area', 'Perimeter']:
    df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan
df.loc[df.sample(frac=0.35, random_state=1).index, 'Eccentricity'] = np.nan

# Fill and drop
df['Area'].fillna(df['Area'].mean(), inplace=True)
df['Perimeter'].fillna(df['Perimeter'].median(), inplace=True)
df.dropna(subset=['Eccentricity'], inplace=True)

# Encode the target label
label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])

# Outlier detection using Z-score
numeric_cols = df.drop(columns=['Class']).columns
for col in numeric_cols:
    z = np.abs(stats.zscore(df[col]))
    df[col] = np.clip(df[col], df[col].mean() - 3 * df[col].std(), df[col].mean() + 3 * df[col].std())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Class']))
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
X_scaled_df['Class'] = df['Class'].values

# Save the preprocessed dataframe for next notebook
X_scaled_df.to_csv("data/preprocessed_scaled.csv", index=False)
print("Preprocessing complete. Output saved to data/preprocessed_scaled.csv")
