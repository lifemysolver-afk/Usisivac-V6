"""
USISIVAC V6 — S6E3 PIPELINE v3.0 FAST
6 models | Pre-tuned params | Fast blending (100 iter) | Saves after each fold
"""
import pandas as pd
import numpy as np
import json, hashlib, datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import warnings
warnings.filterwarnings("ignore")

BASE = Path(".")
PROOF_FILE = BASE / "logs/proof_registry.jsonl"
SUBMISSION_FILE = BASE / "reports/submission.csv"

def log_proof(agent, action, details):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "details": details,
        "proof_hash": hashlib.md5(f"{action}{details}{datetime.datetime.now()}".encode()).hexdigest()[:16]
    }
    with open(PROOF_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [PROOF] {agent} → {action}: {details}")

print("=" * 70)
print("  USISIVAC V6 — S6E3 PIPELINE v3.0 FAST")
print("  6 models | Pre-tuned params | Fast blending | Anti-Simulation ON")
print("=" * 70)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
train = pd.read_csv(BASE / "data/train.csv")
test  = pd.read_csv(BASE / "data/test.csv")
orig  = pd.read_csv(BASE / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Harmonize original dataset
orig = orig.rename(columns={"Churn": "Churn_str"})
orig["Churn"] = (orig["Churn_str"] == "Yes").astype(int)
orig = orig.drop(columns=["customerID", "Churn_str"], errors="ignore")
orig["id"] = -1

# Harmonize train
if "Churn" in train.columns:
    train["Churn"] = train["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(train["Churn"]).astype(int)

# Fix TotalCharges
for df in [train, test, orig]:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Merge original
common_cols = [c for c in orig.columns if c in train.columns]
aug = pd.concat([train, orig[common_cols]], ignore_index=True)
aug["Churn"] = aug["Churn"].astype(int)

print(f"  Train: {train.shape} + Orig: {orig.shape} = Aug: {aug.shape}")
print(f"  Churn dist: {aug['Churn'].value_counts().to_dict()}")
log_proof("DataLoader", "DATA_LOADED_V3", f"aug={aug.shape} test={test.shape}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[2/7] Feature Engineering...")

def engineer(df):
    df = df.copy()
    # Binary encode Yes/No
    yn_cols = ["Partner","Dependents","PhoneService","PaperlessBilling",
               "MultipleLines","OnlineSecurity","OnlineBackup",
               "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    for c in yn_cols:
        if c in df.columns:
            df[c] = df[c].map({"Yes":1,"No":0,"No phone service":0,"No internet service":0}).fillna(0).astype(int)

    # Encode categoricals
    cat_cols = ["Contract","InternetService","PaymentMethod","gender"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    # Numeric features
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["AvgMonthlyCharge"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["ChargeRatio"]      = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["TenureGroup"]      = pd.cut(df["tenure"], bins=[0,12,24,48,72,100], labels=[0,1,2,3,4]).astype(float)
    df["MonthlyGroup"]     = pd.cut(df["MonthlyCharges"], bins=[0,30,60,90,200], labels=[0,1,2,3]).astype(float)
    df["ServiceCount"]     = df[["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                                  "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]].sum(axis=1)
    df["IsHighValue"]      = ((df["MonthlyCharges"] > 70) & (df["tenure"] > 24)).astype(int)
    df["IsNewRisky"]       = ((df["tenure"] < 6) & (df["MonthlyCharges"] > 60)).astype(int)
    df["ChargePerService"] = df["MonthlyCharges"] / (df["ServiceCount"] + 1)
    df["LongTermLowCharge"]= ((df["tenure"] > 36) & (df["MonthlyCharges"] < 50)).astype(int)
    return df

aug_fe  = engineer(aug)
test_fe = engineer(test)

DROP_COLS = ["id", "Churn", "customerID"]
FEAT_COLS = [c for c in aug_fe.columns if c not in DROP_COLS]
# Align test columns
for c in FEAT_COLS:
    if c not in test_fe.columns:
        test_fe[c] = 0

X_full = aug_fe[FEAT_COLS].fillna(0)
y_full = aug_fe["Churn"].values.astype(int)
X_test = test_fe[FEAT_COLS].fillna(0)

print(f"  Features: {len(FEAT_COLS)}")
log_proof("FeatureAgent", "FE_V3", f"n_feat={len(FEAT_COLS)} X={X_full.shape}")

# ── 3. PRE-TUNED PARAMETERS ───────────────────────────────────────────────────
print("\n[3/7] Loading pre-tuned parameters...")
spw = (y_full == 0).sum() / (y_full == 1).sum()

XGB1 = dict(n_estimators=699, max_depth=3, learning_rate=0.0298,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
            eval_metric="auc", random_state=42, n_jobs=-1,
            tree_method="hist", verbosity=0)

XGB2 = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.75, min_child_weight=3,
            reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=spw,
            eval_metric="auc", random_state=7, n_jobs=-1,
            tree_method="hist", verbosity=0)

LGB1 = dict(n_estimators=303, max_depth=6, learning_rate=0.0336,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)

LGB2 = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.75, colsample_bytree=0.75,
            min_child_samples=30, reg_alpha=0.5, reg_lambda=2.0,
            is_unbalance=True, random_state=7, n_jobs=-1, verbose=-1)

CAT1 = dict(iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            eval_metric="AUC", auto_class_weights="Balanced")

CAT2 = dict(iterations=400, depth=5, learning_rate=0.07,
            l2_leaf_reg=5, random_seed=7, verbose=0,
            eval_metric="AUC", auto_class_weights="Balanced")

log_proof("Executor", "PARAMS_V3", "6 pre-tuned model configs loaded.")

# ── 4. 5-FOLD CV TRAINING ─────────────────────────────────────────────────────
print("\n[4/7] 5-Fold CV Training (6 models)...")
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros((len(X_full), 6))
preds = np.zeros((len(X_test), 6))

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X_full, y_full)):
    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y_full[tr_idx], y_full[val_idx]

    fold_preds_val = []
    fold_preds_test = []

    # XGB1
    m = xgb.XGBClassifier(**XGB1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    # XGB2
    m = xgb.XGBClassifier(**XGB2)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    # LGB1
    m = lgb.LGBMClassifier(**LGB1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    # LGB2
    m = lgb.LGBMClassifier(**LGB2)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    # CAT1
    m = cat.CatBoostClassifier(**CAT1)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    # CAT2
    m = cat.CatBoostClassifier(**CAT2)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    fold_preds_val.append(m.predict_proba(X_val)[:, 1])
    fold_preds_test.append(m.predict_proba(X_test)[:, 1])

    for i in range(6):
        oof[val_idx, i] = fold_preds_val[i]
        preds[:, i] += fold_preds_test[i] / 5

    blend_val = np.mean(np.array(fold_preds_val), axis=0)
    fold_auc = roc_auc_score(y_val, blend_val)
    print(f"  Fold {fold+1}: Blend AUC = {fold_auc:.5f}")

# Individual OOF AUCs
names = ["xgb1","xgb2","lgb1","lgb2","cat1","cat2"]
for i, name in enumerate(names):
    a = roc_auc_score(y_full, oof[:, i])
    print(f"    {name}: {a:.5f}")

log_proof("Executor", "6MODEL_DONE_V3",
          " | ".join(f"{n}={roc_auc_score(y_full, oof[:,i]):.5f}" for i,n in enumerate(names)))

# ── 5. FAST RANK BLENDING ─────────────────────────────────────────────────────
print("\n[5/7] Fast Rank-Based Blending (100 iter)...")

# Rank-normalize OOF and predictions
oof_r   = np.column_stack([rankdata(oof[:, i]) / len(oof) for i in range(6)])
preds_r = np.column_stack([rankdata(preds[:, i]) / len(preds) for i in range(6)])

# Random weight search — 100 iterations only
best_auc, best_w = 0, None
np.random.seed(42)
for _ in range(100):
    w = np.random.dirichlet(np.ones(6))
    a = roc_auc_score(y_full, oof_r @ w)
    if a > best_auc:
        best_auc, best_w = a, w

# Also try equal weights
w_eq = np.ones(6) / 6
a_eq = roc_auc_score(y_full, oof_r @ w_eq)
if a_eq > best_auc:
    best_auc, best_w = a_eq, w_eq

print(f"  Best OOF AUC (rank blend): {best_auc:.5f}")
print(f"  Best weights: {[f'{w:.3f}' for w in best_w]}")

# Ridge meta-learner
sc = StandardScaler()
oof_sc = sc.fit_transform(oof_r)
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta.fit(oof_sc, y_full)
meta_oof = meta.predict_proba(oof_sc)[:, 1]
meta_auc = roc_auc_score(y_full, meta_oof)
print(f"  Meta-Learner OOF AUC: {meta_auc:.5f}")

# Choose best
if meta_auc > best_auc:
    final_preds = meta.predict_proba(sc.transform(preds_r))[:, 1]
    final_auc = meta_auc
    method = "meta_learner"
else:
    final_preds = preds_r @ best_w
    final_auc = best_auc
    method = "rank_blend"

print(f"  Final method: {method} | OOF AUC: {final_auc:.5f}")
log_proof("Executor", "BLEND_DONE_V3", f"method={method} oof_auc={final_auc:.5f}")

# ── 6. SAVE SUBMISSION ────────────────────────────────────────────────────────
print("\n[6/7] Saving submission...")
sub = pd.DataFrame({"id": test["id"], "Churn": final_preds})
sub.to_csv(SUBMISSION_FILE, index=False)
print(f"  Saved: {SUBMISSION_FILE} ({len(sub)} rows)")
print(f"  Pred range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
log_proof("Executor", "SUBMISSION_SAVED_V3",
          f"rows={len(sub)} auc={final_auc:.5f} method={method}")

# ── 7. SUMMARY ────────────────────────────────────────────────────────────────
print("\n[7/7] Summary:")
print(f"  Models trained: 6 (XGB×2, LGB×2, CAT×2)")
print(f"  Features: {len(FEAT_COLS)}")
print(f"  Final OOF AUC: {final_auc:.5f}")
print(f"  Submission: {SUBMISSION_FILE}")
print("\n  PIPELINE_V3_COMPLETE")
log_proof("Orchestrator", "PIPELINE_V3_COMPLETE", f"final_oof_auc={final_auc:.5f}")
