import kagglehub
import numpy as np
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering, KMeans
from conf import evaluate_clustering, divisive_clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Download dataset and set up path
path = kagglehub.dataset_download("meharshanali/student-dropout-prediction-dataset")
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))

# =============================================================================
# 2. Data Cleansing (must be done BEFORE X/y split)
# =============================================================================

# 2a. Duplicate check + drop
dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicates removed: {dupes}")

# 2b. Drop missing values with logging
before = len(df)
df = df.dropna()
after = len(df)
print(f"Rows after dropna: {after} ({after / before * 100:.1f}% kept, dropped {before - after})")

# 2c. Binary encoding
df['Internet_Access'] = df['Internet_Access'].map({'No': 0, 'Yes': 1})
df['Part_Time_Job'] = df['Part_Time_Job'].map({'No': 0, 'Yes': 1})
df['Scholarship'] = df['Scholarship'].map({'No': 0, 'Yes': 1})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Validate encoding produced no NaN (catches unexpected values like 'YES', 'yes')
binary_cols = ['Internet_Access', 'Part_Time_Job', 'Scholarship', 'Gender']
nan_after = df[binary_cols].isnull().sum()
if nan_after.any():
    print(f"WARNING — NaN after binary encoding:\n{nan_after[nan_after > 0]}")
else:
    print("Binary encoding: OK (no NaN)")

# 2d. Semester → ordinal integer (Year 1–4 has natural order, one-hot loses it)
semester_map = {'Year 1': 1, 'Year 2': 2, 'Year 3': 3, 'Year 4': 4}
df['Semester'] = df['Semester'].map(semester_map)

# 2e. One-hot encode only nominal columns (Semester excluded now)
df = pd.get_dummies(df, columns=['Parental_Education', 'Department'])

# 2f. Drop Student_ID (identifier, not a feature)
df = df.drop(columns=['Student_ID'])

# 2g. Stress_Index: clip to valid survey range [1, 10]
df['Stress_Index'] = df['Stress_Index'].clip(1, 10)

# 2h. Attendance_Rate: clip to valid percentage range [0, 100]
df['Attendance_Rate'] = df['Attendance_Rate'].clip(0, 100)

# 2i. Family_Income: log1p transform (right-skewed, large values dominate KNN distance)
df['Family_Income'] = np.log1p(df['Family_Income'])

# 2j. IQR outlier removal on all numeric columns
numeric_cols = [
    'Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate',
    'Assignment_Delay_Days', 'Travel_Time_Minutes', 'Stress_Index',
    'GPA', 'Semester_GPA', 'CGPA'
]
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df[numeric_cols] < Q1 - 1.5 * IQR) | (df[numeric_cols] > Q3 + 1.5 * IQR))
print(f"\nOutliers per column:\n{outlier_mask.sum()}")
before_outlier = len(df)
df = df[~outlier_mask.any(axis=1)]
print(f"Rows after outlier removal: {len(df)} (removed {before_outlier - len(df)})")

# 2k. Class distribution (check imbalance before ADASYN)
print(f"\nClass distribution:\n{df['Dropout'].value_counts()}")
print(f"Dropout rate: {df['Dropout'].mean() * 100:.1f}%")

# =============================================================================
# 2l. Feature Engineering (done on clean df, before X/y split)
# =============================================================================

# GPA Trend: positive = improving, negative = declining across semesters
df['GPA_Trend'] = df['Semester_GPA'] - df['CGPA']

# Study Efficiency: GPA earned per study hour — low = struggling despite effort
df['Study_Efficiency'] = df['GPA'] / (df['Study_Hours_per_Day'] + 1e-5)

# Risk Score: combines stress, delays, and poor attendance into one signal
# Higher = more at-risk of dropout
df['Risk_Score'] = (df['Stress_Index'] + df['Assignment_Delay_Days']) - (df['Attendance_Rate'] / 10)

# Financial Pressure: part-time job + no scholarship + lower income = pressure
df['Financial_Pressure'] = df['Part_Time_Job'] + (1 - df['Scholarship']) + (1 / (df['Family_Income'] + 1e-5))

# Engagement Score: high attendance + high study hours = engaged student
df['Engagement_Score'] = df['Attendance_Rate'] * df['Study_Hours_per_Day']

print(f"\nFeature Engineering: added 5 new features "
      f"(GPA_Trend, Study_Efficiency, Risk_Score, Financial_Pressure, Engagement_Score)")

# =============================================================================
# 3. Feature / Target split
# =============================================================================
y = df['Dropout']
X = df.drop(columns=['Dropout'])

# 2l. High-correlation feature drop (threshold > 0.90) — handles GPA/Semester_GPA/CGPA
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [col for col in upper.columns if any(upper[col] > 0.90)]
print(f"\nHigh-correlation features dropped: {high_corr}")
X = X.drop(columns=high_corr)

# Scale → PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Higher k = smaller, purer dropout cluster = better precision
N_CLUSTERS = 8

agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
agg_labels = agg.fit_predict(X_pca)

div_labels = divisive_clustering(X_pca, max_clusters=N_CLUSTERS)

# ── Compare both with silhouette score ───────────────────────
from sklearn.metrics import (silhouette_score, accuracy_score,
                             precision_score, recall_score, f1_score, fbeta_score)

print(f"\nAgglomerative silhouette: {silhouette_score(X_pca, agg_labels):.4f}")
print(f"Divisive      silhouette: {silhouette_score(X_pca, div_labels):.4f}")

# Evaluate both methods
res_agg = evaluate_clustering(agg_labels, "Agglomerative (Bottom-Up, Ward)", y)
res_div = evaluate_clustering(div_labels, "Divisive (Top-Down, KMeans)", y)

# Summary comparison
print(f"\n{'='*65}\nFinal Comparison\n{'='*65}")
print(f"{'Metric':<14}{'Agglomerative':<18}{'Divisive':<18}")
print(f"{'-'*14}{'-'*18}{'-'*18}")
print(f"{'Accuracy':<14}{res_agg['acc']*100:>8.2f}%{'':<10}{res_div['acc']*100:>8.2f}%")
print(f"{'Precision':<14}{res_agg['prec']:>10.4f}{'':<10}{res_div['prec']:>10.4f}")
print(f"{'Recall':<14}{res_agg['rec']:>10.4f}{'':<10}{res_div['rec']:>10.4f}")
print(f"{'F1':<14}{res_agg['f1']:>10.4f}{'':<10}{res_div['f1']:>10.4f}")
print(f"{'F-beta(0.5)':<14}{res_agg['fbeta']:>10.4f}{'':<10}{res_div['fbeta']:>10.4f}")
winner = 'Agglomerative' if res_agg['fbeta'] > res_div['fbeta'] else 'Divisive'
print(f"\nWinner (by F-beta 0.5): {winner}")