# 🚀 The Ultimate Scikit-Learn Cheat Sheet (Modern ML & Kaggle Edition)

Panduan contekan (*cheat sheet*) lengkap dan praktis untuk **Scikit-Learn (versi modern 1.4+)** dari persiapan data, rekayasa fitur, pemodelan, validasi silang, hingga ensembling.

---

## 📑 Daftar Isi
1. [Workflow & Best Practices](#1-workflow--best-practices)
2. [Data Splitting & Cross-Validation](#2-data-splitting--cross-validation)
3. [Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)
4. [ColumnTransformer & Pipeline](#4-columntransformer--pipeline)
5. [Supervised Learning Models](#5-supervised-learning-models)
6. [Unsupervised Learning & Reduksi Dimensi](#6-unsupervised-learning--reduksi-dimensi)
7. [Evaluasi Metrik & Validasi](#7-evaluasi-metrik--validasi)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Ensemble, Stacking & Kalibrasi](#9-ensemble-stacking--kalibrasi)

---

## 1. Workflow & Best Practices

```python
import sklearn
from sklearn import set_config

# 💡 PRO TIP: Mengembalikan output Transformer sebagai Pandas DataFrame (Bukan NumPy Array)
set_config(transform_output="pandas")
```

---

## 2. Data Splitting & Cross-Validation

```python
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold
)

# 1. Train-Test Split Sederhana
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify untuk klasifikasi
)

# 2. K-Fold CV (Untuk Data Regresi)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Stratified K-Fold CV (Wajib untuk Klasifikasi / Kaggle Imbalanced)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Time Series Split (Untuk Data Runtun Waktu / Finansial)
tscv = TimeSeriesSplit(n_splits=5)

# 5. Group K-Fold (Mencegah Data Leakage jika ada ID Kelompok/Pasien/User)
gkf = GroupKFold(n_splits=5)
```

---

## 3. Data Preprocessing & Feature Engineering

### A. Scaling & Transformasi Numerik
```python
from sklearn.preprocessing import (
    StandardScaler,      # Z-score (mean=0, std=1)
    MinMaxScaler,        # Range [0, 1]
    RobustScaler,        # Kebal terhadap Outlier (median & IQR)
    PowerTransformer,    # Menormalkan distribusi skewed (Yeo-Johnson)
    QuantileTransformer  # Mengubah ke distribusi Uniform / Normal
)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Jangan fit ulang pada test data!
```

### B. Encoding Kategorikal
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import TargetEncoder  # ✨ Fitur baru di Scikit-Learn 1.3+

# 1. One-Hot Encoding (Fitur Kategorikal Nominal / Kategori Sedikit)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 2. Ordinal Encoding (Fitur Berurutan: SD, SMP, SMA)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 3. Target Encoding (Sangat ampuh untuk Kategori Kardinalitas Tinggi di Kaggle)
te = TargetEncoder(smooth='auto', cv=5, random_state=42)
X_train['city_encoded'] = te.fit_transform(X_train[['city']], y_train)
X_test['city_encoded'] = te.transform(X_test[['city']])
```

### C. Imputasi Missing Values
```python
from sklearn.impute import SimpleImputer, KNNImputer

# 1. Imputasi Statistik
imp_num = SimpleImputer(strategy='median')        # 'mean', 'median', 'constant'
imp_cat = SimpleImputer(strategy='most_frequent') # Untuk kategorikal / modus

# 2. Imputasi Berbasis Jarak (KNN)
knn_imp = KNNImputer(n_neighbors=5)
```

---

## 4. ColumnTransformer & Pipeline

> 💡 **Best Practice:** Selalu gunakan Pipeline untuk menghindari *Data Leakage*.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Tentukan Kolom
num_cols = ['age', 'income', 'screen_time']
cat_cols = ['gender', 'city', 'education']

# 2. Sub-Pipeline per Tipe Data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. Gabungkan dalam ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ],
    remainder='drop'  # drop kolom lain yang tidak terdaftar
)

# 4. Master End-to-End Pipeline
full_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

---

## 5. Supervised Learning Models

### A. Klasifikasi (*Classification*)
```python
# Linear Models
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Tree-Based & Ensembles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier  # ⚡ Sangat Cepat & mirip LightGBM bawaan Sklearn!
)

# Support Vector & Jarak
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Inisialisasi Model Modern Cepat
model_clf = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.03,
    early_stopping=True,
    random_state=42
)
```

### B. Regresi (*Regression*)
```python
# Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Ensembles
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

# Regresi Ridge dengan Regularisasi L2
model_reg = Ridge(alpha=1.0)
```

---

## 6. Unsupervised Learning & Reduksi Dimensi

```python
# 1. Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

# 2. Reduksi Dimensi (Dimensionality Reduction)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## 7. Evaluasi Metrik & Validasi

### A. Metrik Klasifikasi
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,       # Wajib probabilitas: model.predict_proba(X)[:, 1]
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

# Contoh Penggunaan AUC & LogLoss
y_prob = full_pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
loss = log_loss(y_test, y_prob)

print(f"ROC-AUC: {auc:.5f} | LogLoss: {loss:.5f}")
print(classification_report(y_test, full_pipeline.predict(X_test)))
```

### B. Metrik Regresi
```python
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,  # ✨ Tersedia di Sklearn 1.4+
    mean_absolute_error,
    r2_score
)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 8. Hyperparameter Tuning

### A. RandomizedSearchCV (Efisien & Cepat)
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'model__max_iter': [100, 300, 500, 1000],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'model__max_leaf_nodes': [15, 31, 63, 127],
}

search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

print(f"Best Score: {search.best_score_:.5f}")
print(f"Best Params: {search.best_params_}")
```

---

## 9. Ensemble, Stacking & Kalibrasi

### A. Voting Classifier (Blending Sederhana)
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('hist_gb', HistGradientBoostingClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # 'soft' merata-ratakan probabilitas
)
```

### B. Stacking (Meta-Model Layer 2)
```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('hist_gb', HistGradientBoostingClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
```

### C. Kalibrasi Probabilitas (CalibratedClassifierCV)
```python
from sklearn.calibration import CalibratedClassifierCV

# Mengkalibrasi probabilitas agar tidak overconfident
calibrated_clf = CalibratedClassifierCV(estimator=full_pipeline, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)
```

---

## 🏆 Template Struktur 5-Fold Cross-Validation Standar Kaggle

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.5f}")

overall_auc = roc_auc_score(y_train, oof_preds)
print(f"⭐ Overall OOF ROC-AUC: {overall_auc:.5f}")
```
