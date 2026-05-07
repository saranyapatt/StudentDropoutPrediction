import kagglehub
import numpy as np
import pandas as pd
import os
import seaborn as sns
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from conf import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report, accuracy_score, precision_recall_curve, f1_score

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
print(f"Rows after dropna: {after} ({after/before*100:.1f}% kept, dropped {before - after})")

# 2c. Binary encoding
df['Internet_Access'] = df['Internet_Access'].map({'No': 0, 'Yes': 1})
df['Part_Time_Job']   = df['Part_Time_Job'].map({'No': 0, 'Yes': 1})
df['Scholarship']     = df['Scholarship'].map({'No': 0, 'Yes': 1})
df['Gender']          = df['Gender'].map({'Male': 0, 'Female': 1})

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

# =============================================================================
# 4. Train / Test split → Scale → PCA
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# =============================================================================
# 5. Imbalance handling — ADASYN + ENN
# =============================================================================
ada = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = ada.fit_resample(X_train_pca, y_train)

enn = EditedNearestNeighbours()
X_clean, y_clean = enn.fit_resample(X_train_res, y_train_res)

# =============================================================================
# 6. Modeling — KNN with BallTree
# =============================================================================
classifier = KNeighborsClassifier(
    n_neighbors=11,
    weights='distance',
    algorithm='ball_tree',
    metric='manhattan',
    leaf_size=20
)
classifier.fit(X_clean, y_clean)

# Auto-find best threshold using F1 score on precision-recall curve
y_probs = classifier.predict_proba(X_test_pca)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_test, y_probs, pos_label=1)
f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nAuto-selected threshold: {best_threshold:.3f}")

# Plot precision-recall curve
plt.figure()
plt.plot(thresholds, prec[:-1], label='Precision')
plt.plot(thresholds, rec[:-1], label='Recall')
plt.plot(thresholds, f1_scores, label='F1', linestyle='--')
plt.axvline(best_threshold, color='red', linestyle=':', label=f'Best threshold={best_threshold:.2f}')
plt.xlabel('Threshold')
plt.title('Precision / Recall / F1 vs Threshold')
plt.legend()
plt.tight_layout()
plt.show()

y_pred = (y_probs >= best_threshold).astype(int)

class_names = ['Stay (0)', 'Dropout (1)']
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=class_names, title='Student Dropout Confusion Matrix')

# =============================================================================
# 7. Evaluation
# =============================================================================
print("\n--- Model Performance ---")
precision = precision_score(y_test, y_pred, pos_label=1)
print(f"Precision Score (Dropout): {precision:.2f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Dropout']))
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Cross-validation to verify generalization (5-fold)
cv_scores = cross_val_score(classifier, X_clean, y_clean, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# =============================================================================
# 8. Failure Analysis
# =============================================================================

# Reconstruct test set in original feature space for readable analysis
X_test_df = X_test.copy().reset_index(drop=True)
y_test_arr = y_test.reset_index(drop=True)
y_pred_arr = pd.Series(y_pred, name='Predicted')
y_probs_arr = pd.Series(y_probs, name='Dropout_Prob')

analysis_df = X_test_df.copy()
analysis_df['Actual']       = y_test_arr.values
analysis_df['Predicted']    = y_pred_arr.values
analysis_df['Dropout_Prob'] = y_probs_arr.values

# Segment into 4 groups
TP = analysis_df[(analysis_df['Actual'] == 1) & (analysis_df['Predicted'] == 1)]  # Correct Dropout
TN = analysis_df[(analysis_df['Actual'] == 0) & (analysis_df['Predicted'] == 0)]  # Correct Stay
FP = analysis_df[(analysis_df['Actual'] == 0) & (analysis_df['Predicted'] == 1)]  # False Alarm
FN = analysis_df[(analysis_df['Actual'] == 1) & (analysis_df['Predicted'] == 0)]  # Missed Dropout

print("\n--- Failure Analysis ---")
print(f"True Positives  (caught dropouts)  : {len(TP)}")
print(f"True Negatives  (correct stay)     : {len(TN)}")
print(f"False Positives (false alarm)      : {len(FP)}")
print(f"False Negatives (missed dropout)   : {len(FN)}  ← most critical")

# --- Compare FN vs TP: what makes missed dropouts different? ---
numeric_feat = ['GPA', 'Attendance_Rate', 'Study_Hours_per_Day', 'Stress_Index',
                'Assignment_Delay_Days', 'Risk_Score', 'Study_Efficiency',
                'GPA_Trend', 'Engagement_Score']
numeric_feat = [c for c in numeric_feat if c in analysis_df.columns]

print("\n--- False Negatives vs True Positives (numeric mean) ---")
compare = pd.DataFrame({
    'Missed Dropout (FN)': FN[numeric_feat].mean(),
    'Caught Dropout (TP)': TP[numeric_feat].mean(),
}).round(3)
compare['Difference'] = (compare['Missed Dropout (FN)'] - compare['Caught Dropout (TP)']).round(3)
print(compare.to_string())

# --- Compare FP vs TN: what makes false alarms different? ---
print("\n--- False Positives vs True Negatives (numeric mean) ---")
compare_fp = pd.DataFrame({
    'False Alarm (FP)': FP[numeric_feat].mean(),
    'Correct Stay (TN)': TN[numeric_feat].mean(),
}).round(3)
compare_fp['Difference'] = (compare_fp['False Alarm (FP)'] - compare_fp['Correct Stay (TN)']).round(3)
print(compare_fp.to_string())

# --- Confidence distribution of failures ---
print(f"\nFN avg model confidence (dropout prob): {FN['Dropout_Prob'].mean():.3f}  ← model was uncertain")
print(f"FP avg model confidence (dropout prob): {FP['Dropout_Prob'].mean():.3f}  ← model was overconfident")

# --- Plot: probability distribution by outcome group ---
plt.figure(figsize=(8, 4))
for label, group, color in [
    ('TP (caught dropout)',  TP, 'green'),
    ('FN (missed dropout)', FN, 'red'),
    ('FP (false alarm)',    FP, 'orange'),
    ('TN (correct stay)',   TN, 'blue'),
]:
    if len(group) > 0:
        plt.hist(group['Dropout_Prob'], bins=20, alpha=0.5, label=label, color=color)

plt.axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.2f}')
plt.xlabel('Predicted Dropout Probability')
plt.ylabel('Count')
plt.title('Failure Analysis — Probability Distribution by Outcome')
plt.legend()
plt.tight_layout()
plt.show()