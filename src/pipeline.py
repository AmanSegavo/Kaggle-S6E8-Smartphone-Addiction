"""
Kaggle Playground Series S6E8: Predicting Smartphone Addiction
End-to-End Machine Learning Pipeline (LightGBM, XGBoost, CatBoost)
"""

import os
import zipfile
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

SEED = 42
N_SPLITS = 5
TARGET = 'addicted_label'
ID_COL = 'id'


def load_and_preprocess(train_path: str, test_path: str):
    """Load train/test data, apply feature engineering, and encode categorical columns."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    def feature_engineering(df):
        df = df.copy()
        if 'daily_screen_time_hours' in df.columns and 'sleep_hours' in df.columns:
            df['screen_to_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 0.1)
        if 'social_media_hours' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['social_screen_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 0.1)
        if 'weekend_screen_time' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['weekend_screen_diff'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
        if 'notifications_per_day' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['notifications_per_screen_hour'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + 0.1)
        if 'app_opens_per_day' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['opens_per_screen_hour'] = df['app_opens_per_day'] / (df['daily_screen_time_hours'] + 0.1)
        return df

    train = feature_engineering(train)
    test = feature_engineering(test)

    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    features = [c for c in train.columns if c not in [ID_COL, TARGET]]
    return train, test, features


def train_models(train: pd.DataFrame, test: pd.DataFrame, features: list, models_dir: str = 'models'):
    """Train 5-Fold Stratified LightGBM, XGBoost, and CatBoost models."""
    os.makedirs(models_dir, exist_ok=True)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    X_train = train[features]
    y_train = train[TARGET]
    X_test = test[features]

    # --- 1. LightGBM ---
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 127,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': SEED,
    }
    oof_lgb = np.zeros(len(X_train))
    pred_lgb = np.zeros(len(X_test))
    lgb_scores = []

    print('🚀 [1/3] Training LightGBM...')
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = lgb.train(
            lgb_params, lgb.Dataset(X_tr, label=y_tr),
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
        )
        model.save_model(os.path.join(models_dir, f'lgbm_fold{fold+1}.txt'))
        oof_lgb[val_idx] = model.predict(X_val)
        pred_lgb += model.predict(X_test) / N_SPLITS
        score = roc_auc_score(y_val, oof_lgb[val_idx])
        lgb_scores.append(score)

    lgb_oof = roc_auc_score(y_train, oof_lgb)
    print(f'✅ LightGBM OOF AUC : {lgb_oof:.5f}')

    # --- 2. XGBoost ---
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.02,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': SEED,
    }
    oof_xgb = np.zeros(len(X_train))
    pred_xgb = np.zeros(len(X_test))
    xgb_scores = []

    print('\n🚀 [2/3] Training XGBoost (GPU)...')
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=500)
        model.save_model(os.path.join(models_dir, f'xgb_fold{fold+1}.json'))

        oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_xgb += model.predict_proba(X_test)[:, 1] / N_SPLITS
        score = roc_auc_score(y_val, oof_xgb[val_idx])
        xgb_scores.append(score)

    xgb_oof = roc_auc_score(y_train, oof_xgb)
    print(f'✅ XGBoost OOF AUC  : {xgb_oof:.5f}')

    # --- 3. CatBoost ---
    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.02,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU',
        'random_seed': SEED,
        'l2_leaf_reg': 3.0,
        'early_stopping_rounds': 100,
        'verbose': False,
    }
    oof_cb = np.zeros(len(X_train))
    pred_cb = np.zeros(len(X_test))
    cb_scores = []

    print('\n🚀 [3/3] Training CatBoost (GPU)...')
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**cb_params)
        model.fit(Pool(X_tr, label=y_tr), eval_set=Pool(X_val, label=y_val), use_best_model=True)
        model.save_model(os.path.join(models_dir, f'catboost_fold{fold+1}.cbm'))

        oof_cb[val_idx] = model.predict_proba(X_val)[:, 1]
        pred_cb += model.predict_proba(X_test)[:, 1] / N_SPLITS
        score = roc_auc_score(y_val, oof_cb[val_idx])
        cb_scores.append(score)

    cb_oof = roc_auc_score(y_train, oof_cb)
    print(f'✅ CatBoost OOF AUC : {cb_oof:.5f}')

    return (oof_lgb, oof_xgb, oof_cb), (pred_lgb, pred_xgb, pred_cb), (lgb_scores, xgb_scores, cb_scores)


if __name__ == '__main__':
    print("Pipeline ready for execution.")
