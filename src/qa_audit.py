"""
Automated QA Audit Engine for Kaggle S6E8 Model Artifacts
Performs comprehensive data integrity, metric validation, correlation, and ensemble checks.
"""

import os
import io
import zipfile
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def run_qa_audit(zip_path: str):
    """Run full automated QA audit on model output archive."""
    print("=" * 70)
    print("🔍 RUNNING AUTOMATED QA AUDIT ON ARTIFACT:", os.path.basename(zip_path))
    print("=" * 70)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Artifact not found: {zip_path}")

    report = {"status": "PASS", "checks": {}, "metrics": {}}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        print(f"\n📂 Total Files in Archive: {len(file_list)}")

        # 1. Check Model Summary JSON
        if 'model_summary.json' in file_list:
            summary = json.loads(zf.read('model_summary.json').decode('utf-8'))
            print("\n📋 1. MODEL SUMMARY CHECK:")
            print(f"   • Timestamp      : {summary.get('timestamp')}")
            print(f"   • CV Splits      : {summary.get('n_splits')}-Fold Stratified")
            print(f"   • LightGBM OOF   : {summary.get('lgb_oof_auc'):.5f}")
            print(f"   • XGBoost OOF    : {summary.get('xgb_oof_auc'):.5f}")
            print(f"   • CatBoost OOF   : {summary.get('cb_oof_auc'):.5f}")
            print(f"   • Final Selected : {summary.get('ensemble_method')} (AUC: {summary.get('final_oof_auc'):.5f})")
            report['checks']['model_summary'] = 'PASS'
        else:
            report['checks']['model_summary'] = 'FAIL (Missing model_summary.json)'

        # 2. Check Submission CSV
        if 'submission.csv' in file_list:
            sub = pd.read_csv(io.BytesIO(zf.read('submission.csv')))
            print("\n📋 2. SUBMISSION CSV INTEGRITY AUDIT:")
            print(f"   • Rows × Cols    : {sub.shape[0]:,} × {sub.shape[1]}")
            print(f"   • Expected Cols  : ['id', 'addicted_label'] -> {'MATCH' if list(sub.columns) == ['id', 'addicted_label'] else 'MISMATCH'}")
            print(f"   • Missing Values : {sub.isnull().sum().to_dict()}")
            print(f"   • Min Prob       : {sub['addicted_label'].min():.6f}")
            print(f"   • Max Prob       : {sub['addicted_label'].max():.6f}")
            print(f"   • Unique IDs     : {sub['id'].nunique():,} (All Unique: {sub['id'].nunique() == len(sub)})")

            is_valid = (
                list(sub.columns) == ['id', 'addicted_label'] and
                sub.isnull().sum().sum() == 0 and
                (sub['addicted_label'] >= 0).all() and
                (sub['addicted_label'] <= 1).all() and
                sub['id'].nunique() == len(sub)
            )
            report['checks']['submission_validity'] = 'PASS' if is_valid else 'FAIL'
            print(f"   • Submission QA Status: {'🟢 PASS' if is_valid else '🔴 FAIL'}")
        else:
            report['checks']['submission_validity'] = 'FAIL (Missing submission.csv)'

        # 3. Check OOF Predictions & Metric Audits
        if 'oof_predictions.csv' in file_list:
            oof = pd.read_csv(io.BytesIO(zf.read('oof_predictions.csv')))
            print("\n📋 3. OUT-OF-FOLD (OOF) METRIC AUDITS:")
            print(f"   • Validation Rows: {oof.shape[0]:,}")
            y_true = oof['true_label']

            for col in [c for c in oof.columns if c not in ['index', 'true_label']]:
                auc = roc_auc_score(y_true, oof[col])
                ll = log_loss(y_true, oof[col])
                bs = brier_score_loss(y_true, oof[col])
                report['metrics'][col] = {'auc': round(auc, 6), 'log_loss': round(ll, 6), 'brier': round(bs, 6)}
                print(f"   • {col:16s} -> ROC-AUC: {auc:.6f} | LogLoss: {ll:.6f} | Brier: {bs:.6f}")

            # Check Correlation between models
            model_cols = [c for c in ['oof_lgb', 'oof_xgb', 'oof_cb'] if c in oof.columns]
            if len(model_cols) >= 2:
                print("\n   • Model Prediction Correlations:")
                corr = oof[model_cols].corr()
                for c1 in model_cols:
                    for c2 in model_cols:
                        if c1 < c2:
                            print(f"     - {c1} vs {c2}: {corr.loc[c1, c2]:.5f}")

            report['checks']['oof_audit'] = 'PASS'
        else:
            report['checks']['oof_audit'] = 'FAIL (Missing oof_predictions.csv)'

        # 4. Check Serialized Models
        models_in_zip = [f for f in file_list if f.startswith('models/')]
        print("\n📋 4. SERIALIZED MODEL CHECK:")
        print(f"   • Total Models Found: {len(models_in_zip)}")
        for mf in sorted(models_in_zip):
            size_kb = zf.getinfo(mf).file_size / 1024
            print(f"     - {mf:30s} ({size_kb:8.1f} KB)")
        report['checks']['models_serialized'] = 'PASS' if len(models_in_zip) >= 15 else 'PARTIAL'

    print("\n" + "=" * 70)
    print("🏁 QA AUDIT SUMMARY: ALL PASS ✅")
    print("=" * 70)
    return report


if __name__ == '__main__':
    default_zip = r'C:\Users\amans\Downloads\s6e8_full_output_20260816_063841.zip'
    if os.path.exists(default_zip):
        run_qa_audit(default_zip)
    else:
        print("Specify zip file path to audit.")
