# 🎓 Student Dropout Prediction — Supervised vs Unsupervised Learning

A machine learning project that predicts and analyses university student dropout patterns using **two complementary approaches**: a Supervised Learning model (KNN) for individual prediction, and Unsupervised Learning models (Hierarchical Clustering — both Agglomerative and Divisive) for discovering at-risk student groups.

---

## 📋 Project Overview

This project tackles the problem of **early student dropout detection** on an imbalanced dataset, applying:
- Structured data cleansing (11 steps)
- Feature engineering (5 new derived features)
- Class imbalance handling (ADASYN + ENN)
- Both Supervised and Unsupervised models
- Failure analysis on misclassifications

**Dataset:** [Student Dropout Prediction Dataset](https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset) — Kaggle (downloaded automatically via `kagglehub`)

---

## 📁 Project Structure

```
.
├── StudentDropoutPredictionSL.py   # Supervised Learning — KNN with BallTree
├── StudentDropoutPredictionUSL.py  # Unsupervised Learning — Hierarchical Clustering
├── conf.py                          # Shared utility functions
├── presentation.html                # reveal.js presentation slides
└── README.md
```

---

## 🛠️ Requirements

```bash
pip install kagglehub numpy pandas scikit-learn imbalanced-learn matplotlib seaborn scipy
```

**Python version:** 3.9+

---

## 🚀 How to Run

### Supervised Learning (KNN)
```bash
python StudentDropoutPredictionSL.py
```

### Unsupervised Learning (Hierarchical Clustering)
```bash
python StudentDropoutPredictionUSL.py
```

### View Presentation
Open `presentation.html` in any modern browser.

---

## 🧹 Data Cleansing Pipeline (11 Steps)

| # | Step | Purpose |
|---|---|---|
| 1 | Drop duplicates | Prevent inflated training data |
| 2 | Drop missing values (with logging) | Track data loss |
| 3 | Binary encoding (Yes/No → 1/0) + NaN validation | Catch unexpected values silently becoming NaN |
| 4 | Convert Semester (Year 1–4) to ordinal int | Preserve natural ordering |
| 5 | One-hot encode Department & Parental_Education | Handle nominal categories |
| 6 | Drop Student_ID | Identifier, not predictive |
| 7 | Clip Stress_Index to [1, 10] | Remove invalid survey responses |
| 8 | Clip Attendance_Rate to [0, 100] | Percentage validation |
| 9 | log1p transform on Family_Income | Handle right-skewed distribution |
| 10 | IQR outlier removal on numeric columns | Reduce noise for distance-based models |
| 11 | Drop highly correlated features (>0.90) | Remove redundancy among GPA/Semester_GPA/CGPA |

---

## ⚙️ Feature Engineering — 5 New Features

| Feature | Formula | Purpose |
|---|---|---|
| `GPA_Trend` | `Semester_GPA − CGPA` | Detect declining performance |
| `Study_Efficiency` | `GPA / Study_Hours_per_Day` | Identify struggling students |
| `Risk_Score` | `Stress + Delays − Attendance/10` | Composite risk signal |
| `Financial_Pressure` | `Part_Time_Job + (1 − Scholarship) + 1/Income` | Financial stress indicator |
| `Engagement_Score` | `Attendance × Study_Hours` | Multiplicative engagement signal |

---

## 🤖 Supervised Learning — KNN Model

**Configuration:**
```python
KNeighborsClassifier(
    n_neighbors=11,
    weights='distance',
    algorithm='ball_tree',
    metric='manhattan',
    leaf_size=20
)
```

**Pipeline:**
1. Train/test split (85/15, stratified)
2. StandardScaler → PCA(0.95)
3. ADASYN oversampling + ENN cleaning
4. KNN training
5. Auto-tuned threshold via F1-optimized precision-recall curve
6. Evaluation: confusion matrix, precision, recall, F1, accuracy
7. Failure analysis on FN vs TP groups

---

## 🔍 Unsupervised Learning — Hierarchical Clustering

Two approaches implemented and compared:

### 1. Agglomerative (Bottom-Up)
- Each student starts as its own cluster
- Merges most similar pairs using **Ward linkage** (minimizes within-cluster variance)
- Continues until k clusters remain

### 2. Divisive (Top-Down)
- All students start in one cluster
- Recursively split using KMeans into the largest cluster
- Continues until k clusters formed

**Pipeline:**
1. StandardScaler → PCA(0.95) on full dataset
2. Run both clustering methods with k=8
3. Map clusters to labels using auto-tuned threshold (optimized via F-beta(0.5), which weights precision 2× more than recall)
4. Evaluate using:
   - Silhouette score (cluster separation quality)
   - Confusion matrix
   - Precision, Recall, F1, F-beta(0.5)
5. Visualize clusters in 2D PCA space

---

## 📊 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Precision** | Of all predicted dropouts, how many were correct? |
| **Recall** | Of all real dropouts, how many did we catch? |
| **F1 Score** | Harmonic mean of precision and recall |
| **F-beta(0.5)** | Weighted F1 with 2× more emphasis on precision |
| **Silhouette Score** | Cluster cohesion (USL only, range −1 to +1) |

---

## 🔬 Failure Analysis

The SL model is analysed on 4 outcome types:

| Outcome | Description | Priority |
|---|---|---|
| ✅ TP | Real dropout → predicted dropout | — |
| ✅ TN | Real stay → predicted stay | — |
| 🟠 FP | Real stay → predicted dropout | Medium |
| 🔴 FN | Real dropout → predicted stay | **Critical** |

The analysis compares feature distributions of FN vs TP students to understand why the model misses certain dropouts.

---

## 🔮 Future Improvements

- **GridSearchCV** to systematically tune KNN hyperparameters
- **XGBoost / Random Forest** for potentially stronger tabular performance
- **Voting Ensemble** combining KNN + Tree-based models
- **sklearn Pipeline** to prevent data leakage in cross-validation
- **Additional features** — counsellor notes, social participation, library usage

---

## 🧠 Key Takeaways

1. **Cleansing matters** — KNN distance is highly sensitive to outliers and feature scale
2. **Imbalance must be addressed** — without ADASYN+ENN, the model defaults to predicting "Stay"
3. **Feature engineering reveals hidden signals** — composite features captured patterns raw data couldn't
4. **Threshold tuning is critical** — auto-optimization via F-beta avoids manual guessing and bias
5. **SL + USL together is more powerful** — clustering finds at-risk groups, classification flags individuals

---

## 📜 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Dataset: [meharshanali on Kaggle](https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset)
- Built with `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`
