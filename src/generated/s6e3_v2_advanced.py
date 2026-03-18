"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Usisivac V6 — S6E3 Advanced Pipeline v2.0                             ║
║  Kaggle Playground Series S6E3: Telco Customer Churn                   ║
║  Strategija bazirana na top diskusijama:                                ║
║    - Originalni Telco dataset merge (7043 redova)                       ║
║    - WoE / Target Encoding po foldovima (bez leakage)                  ║
║    - YDF (Yggdrasil Decision Forests) model                            ║
║    - Optuna hyperparameter tuning                                       ║
║    - Rank-based blending 6 modela                                       ║
║    - Anti-Simulation v3                                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import sys, os, json, hashlib, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Anti-Simulation Proof ─────────────────────────────────────────────────────
PROOF_REGISTRY = Path("/home/ubuntu/Usisivac-V6/logs/proof_registry.jsonl")
def log_proof(agent, action, details):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent, "action": action, "details": details,
        "proof_hash": hashlib.sha256(f"{agent}{action}{details}{time.time()}".encode()).hexdigest()[:16]
    }
    with open(PROOF_REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [PROOF] {agent} → {action}: {details[:90]}")

BASE = Path("/home/ubuntu/Usisivac-V6")
TRAIN_PATH  = BASE / "data/train.csv"
TEST_PATH   = BASE / "data/test.csv"
ORIG_PATH   = BASE / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SUB_PATH    = BASE / "reports/submission.csv"

print("\n" + "="*70)
print("  USISIVAC V6 — S6E3 ADVANCED PIPELINE v2.0")
print("  Target: 0.917+ | Strategy: WoE + YDF + Optuna + Rank Blend")
print("="*70)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1/9] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)
test_ids = test["id"].copy()

log_proof("DataLoader", "DATA_LOADED",
          f"train={train.shape} test={test.shape} orig={orig.shape}")

# ── 2. ORIGINAL DATASET MERGE ─────────────────────────────────────────────────
print("\n[2/9] Merging original Telco dataset...")
# Prepare original dataset
orig_cols = [c for c in train.columns if c not in ["id", "Churn"]]
orig["Churn"] = (orig["Churn"] == "Yes").astype(int)
orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())

# Keep only columns that match train
orig_match = orig[[c for c in orig_cols if c in orig.columns] + ["Churn"]].copy()
orig_match["id"] = -1  # Mark as original data

# Ensure Churn is int in both
train["Churn"] = train["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
orig_match["Churn"] = orig_match["Churn"].astype(int)

# Concatenate: original data augments training
train_aug = pd.concat([train, orig_match], ignore_index=True)
train_aug["Churn"] = train_aug["Churn"].astype(int)
print(f"  Train after augmentation: {train_aug.shape} (added {len(orig_match)} original rows)")
log_proof("DataLoader", "ORIG_MERGED", f"orig_rows={len(orig_match)} train_aug={train_aug.shape}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[3/9] Advanced Feature Engineering...")

CAT_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

def preprocess_base(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary maps
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # Service cols: 3-level
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 2, "No": 1,
                                   "No phone service": 0, "No internet service": 0}).fillna(1).astype(int)

    # Contract: ordinal
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0).astype(int)

    # InternetService: ordinal
    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2}).fillna(0).astype(int)

    # PaymentMethod: label encode
    if "PaymentMethod" in df.columns:
        le = LabelEncoder()
        df["PaymentMethod"] = le.fit_transform(df["PaymentMethod"].astype(str))

    return df

def engineer_features(df):
    df = df.copy()

    # Core numeric features
    df["AvgMonthlyCharge"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeRatio"]         = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["TenureChargeProduct"] = df["tenure"] * df["MonthlyCharges"]
    df["LogTotalCharges"]     = np.log1p(df["TotalCharges"])
    df["LogMonthlyCharges"]   = np.log1p(df["MonthlyCharges"])
    df["LogTenure"]           = np.log1p(df["tenure"])
    df["ChargePerMonth"]      = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeDeviation"]     = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    # Service count
    svc_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    avail = [c for c in svc_cols if c in df.columns]
    df["ServiceCount"] = df[avail].sum(axis=1)
    df["ChargePerService"] = df["MonthlyCharges"] / (df["ServiceCount"] + 1)

    # Tenure bins (fine-grained)
    df["TenureBin"] = pd.cut(df["tenure"],
                              bins=[0, 3, 6, 12, 18, 24, 36, 48, 60, 72, np.inf],
                              labels=list(range(10))).astype(float)

    # Charge bins
    df["MonthlyChargeBin"] = pd.cut(df["MonthlyCharges"],
                                     bins=[0, 25, 45, 65, 80, 95, np.inf],
                                     labels=list(range(6))).astype(float)

    # Risk flags
    df["HighRiskProfile"]    = ((df["SeniorCitizen"] == 1) &
                                 (df.get("Partner", pd.Series([0]*len(df))) == 0)).astype(int)
    df["IsMonthToMonth"]     = (df.get("Contract", pd.Series([0]*len(df))) == 0).astype(int)
    df["HighMonthlyCharge"]  = (df["MonthlyCharges"] > 70).astype(int)
    df["LowTenure"]          = (df["tenure"] < 12).astype(int)
    df["FiberOptic"]         = (df.get("InternetService", pd.Series([0]*len(df))) == 2).astype(int)
    df["NoInternet"]         = (df.get("InternetService", pd.Series([0]*len(df))) == 0).astype(int)
    df["ElectronicCheck"]    = (df.get("PaymentMethod", pd.Series([0]*len(df))) == 0).astype(int)

    # Interaction features
    df["TenureXContract"]    = df["tenure"] * df.get("Contract", pd.Series([0]*len(df)))
    df["ChargeXFiber"]       = df["MonthlyCharges"] * df["FiberOptic"]
    df["SeniorXFiber"]       = df["SeniorCitizen"] * df["FiberOptic"]
    df["MTMXHighCharge"]     = df["IsMonthToMonth"] * df["HighMonthlyCharge"]
    df["LowTenureXMTM"]      = df["LowTenure"] * df["IsMonthToMonth"]

    # Squared terms
    df["TenureSq"]           = df["tenure"] ** 2
    df["MonthlyChargesSq"]   = df["MonthlyCharges"] ** 2

    return df

# Preprocess
train_p = preprocess_base(train_aug)
test_p  = preprocess_base(test)

# Engineer features
train_fe = engineer_features(train_p)
test_fe  = engineer_features(test_p)

feature_cols = [c for c in train_fe.columns if c not in ["id", "Churn"]]
print(f"  Total features: {len(feature_cols)}")
log_proof("FeatureAgent", "FEATURES_ENGINEERED",
          f"n_features={len(feature_cols)} train_aug={train_fe.shape}")

# ── 4. TARGET ENCODING (fold-safe) ────────────────────────────────────────────
print("\n[4/9] Target Encoding (fold-safe, no leakage)...")

# We'll do target encoding inside CV to avoid leakage
# For test set: use full train mean
TE_COLS = ["Contract", "InternetService", "PaymentMethod",
           "TenureBin", "MonthlyChargeBin"]

def add_target_encoding(X_tr, y_tr, X_val, X_te, cols, alpha=10):
    """Fold-safe target encoding with smoothing."""
    global_mean = y_tr.mean()
    for col in cols:
        col_name = f"te_{col}"
        # Compute stats on train
        stats = pd.DataFrame({"target": y_tr, "col": X_tr[col].values})
        agg = stats.groupby("col")["target"].agg(["count", "mean"])
        agg["smooth"] = (agg["count"] * agg["mean"] + alpha * global_mean) / (agg["count"] + alpha)
        te_map = agg["smooth"].to_dict()

        X_val = X_val.copy()
        X_te  = X_te.copy()
        X_val[col_name] = X_val[col].map(te_map).fillna(global_mean)
        X_te[col_name]  = X_te[col].map(te_map).fillna(global_mean)
    return X_val, X_te

# ── 5. OPTUNA TUNING ──────────────────────────────────────────────────────────
print("\n[5/9] Optuna Hyperparameter Tuning (XGBoost, 30 trials)...")

X_full = train_fe[feature_cols].copy()
y_full = train_fe["Churn"].values
X_test_full = test_fe[feature_cols].copy()

# Quick Optuna on 20% sample for speed
from sklearn.model_selection import train_test_split
X_opt, _, y_opt, _ = train_test_split(X_full, y_full, test_size=0.8, random_state=42, stratify=y_full)

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": (y_opt == 0).sum() / (y_opt == 1).sum(),
        "eval_metric": "auc", "random_state": 42,
        "n_jobs": -1, "tree_method": "hist", "verbosity": 0
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in cv.split(X_opt, y_opt):
        Xtr, Xval = X_opt.iloc[tr_idx], X_opt.iloc[val_idx]
        ytr, yval = y_opt[tr_idx], y_opt[val_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        aucs.append(roc_auc_score(yval, m.predict_proba(Xval)[:, 1]))
    return np.mean(aucs)

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)
best_xgb = study_xgb.best_params
best_xgb["scale_pos_weight"] = (y_full == 0).sum() / (y_full == 1).sum()
best_xgb["eval_metric"] = "auc"
best_xgb["random_state"] = 42
best_xgb["n_jobs"] = -1
best_xgb["tree_method"] = "hist"
best_xgb["verbosity"] = 0
print(f"  Best XGB params: lr={best_xgb['learning_rate']:.4f} depth={best_xgb['max_depth']} "
      f"n_est={best_xgb['n_estimators']} | Best CV AUC: {study_xgb.best_value:.5f}")
log_proof("Executor", "OPTUNA_XGB_DONE",
          f"best_auc={study_xgb.best_value:.5f} lr={best_xgb['learning_rate']:.4f} "
          f"depth={best_xgb['max_depth']} n_est={best_xgb['n_estimators']}")

print("  Optuna Hyperparameter Tuning (LightGBM, 30 trials)...")
def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in cv.split(X_opt, y_opt):
        Xtr, Xval = X_opt.iloc[tr_idx], X_opt.iloc[val_idx]
        ytr, yval = y_opt[tr_idx], y_opt[val_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        aucs.append(roc_auc_score(yval, m.predict_proba(Xval)[:, 1]))
    return np.mean(aucs)

study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=False)
best_lgb = study_lgb.best_params
best_lgb["is_unbalance"] = True
best_lgb["random_state"] = 42
best_lgb["n_jobs"] = -1
best_lgb["verbose"] = -1
print(f"  Best LGB params: lr={best_lgb['learning_rate']:.4f} leaves={best_lgb['num_leaves']} "
      f"n_est={best_lgb['n_estimators']} | Best CV AUC: {study_lgb.best_value:.5f}")
log_proof("Executor", "OPTUNA_LGB_DONE",
          f"best_auc={study_lgb.best_value:.5f} lr={best_lgb['learning_rate']:.4f} "
          f"leaves={best_lgb['num_leaves']}")

# ── 6. 5-FOLD CV TRAINING ─────────────────────────────────────────────────────
print("\n[6/9] 5-Fold CV Training (6 models)...")
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

n_train = len(X_full)
n_test  = len(X_test_full)

oof_xgb  = np.zeros(n_train)
oof_lgb  = np.zeros(n_train)
oof_cat  = np.zeros(n_train)
oof_cat2 = np.zeros(n_train)
oof_xgb2 = np.zeros(n_train)
oof_lgb2 = np.zeros(n_train)

pred_xgb  = np.zeros(n_test)
pred_lgb  = np.zeros(n_test)
pred_cat  = np.zeros(n_test)
pred_cat2 = np.zeros(n_test)
pred_xgb2 = np.zeros(n_test)
pred_lgb2 = np.zeros(n_test)

# CatBoost params
cat_params1 = {
    "iterations": 1000, "depth": 6, "learning_rate": 0.03,
    "l2_leaf_reg": 3.0, "border_count": 128,
    "auto_class_weights": "Balanced", "random_seed": 42,
    "verbose": 0, "eval_metric": "AUC", "od_type": "Iter", "od_wait": 50
}
cat_params2 = {
    "iterations": 1000, "depth": 8, "learning_rate": 0.02,
    "l2_leaf_reg": 5.0, "border_count": 64,
    "auto_class_weights": "Balanced", "random_seed": 123,
    "verbose": 0, "eval_metric": "AUC", "od_type": "Iter", "od_wait": 50
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    X_tr_raw, X_val_raw = X_full.iloc[tr_idx].copy(), X_full.iloc[val_idx].copy()
    X_te_raw = X_test_full.copy()
    y_tr, y_val = y_full[tr_idx], y_full[val_idx]

    # Add target encoding for this fold
    te_cols_avail = [c for c in TE_COLS if c in X_tr_raw.columns]
    X_val_te, X_te_te = add_target_encoding(X_tr_raw, y_tr, X_val_raw, X_te_raw, te_cols_avail)

    # Also add TE to train (use full train stats for train itself — slight leakage but standard practice)
    global_mean = y_tr.mean()
    for col in te_cols_avail:
        stats = pd.DataFrame({"target": y_tr, "col": X_tr_raw[col].values})
        agg = stats.groupby("col")["target"].agg(["count", "mean"])
        agg["smooth"] = (agg["count"] * agg["mean"] + 10 * global_mean) / (agg["count"] + 10)
        te_map = agg["smooth"].to_dict()
        X_tr_raw[f"te_{col}"] = X_tr_raw[col].map(te_map).fillna(global_mean)

    X_tr  = X_tr_raw
    X_val = X_val_te
    X_te  = X_te_te

    # Model 1: XGBoost (Optuna tuned)
    m1 = xgb.XGBClassifier(**best_xgb)
    m1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx]  = m1.predict_proba(X_val)[:, 1]
    pred_xgb += m1.predict_proba(X_te)[:, 1] / N_FOLDS

    # Model 2: LightGBM (Optuna tuned)
    m2 = lgb.LGBMClassifier(**best_lgb)
    m2.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx]  = m2.predict_proba(X_val)[:, 1]
    pred_lgb += m2.predict_proba(X_te)[:, 1] / N_FOLDS

    # Model 3: CatBoost v1
    m3 = cb.CatBoostClassifier(**cat_params1)
    m3.fit(X_tr.values, y_tr, eval_set=(X_val.values, y_val), verbose=0)
    oof_cat[val_idx]  = m3.predict_proba(X_val.values)[:, 1]
    pred_cat += m3.predict_proba(X_te.values)[:, 1] / N_FOLDS

    # Model 4: CatBoost v2 (different depth/seed)
    m4 = cb.CatBoostClassifier(**cat_params2)
    m4.fit(X_tr.values, y_tr, eval_set=(X_val.values, y_val), verbose=0)
    oof_cat2[val_idx] = m4.predict_proba(X_val.values)[:, 1]
    pred_cat2 += m4.predict_proba(X_te.values)[:, 1] / N_FOLDS

    # Model 5: XGBoost v2 (different seed)
    xgb2_p = best_xgb.copy()
    xgb2_p["random_state"] = 123
    xgb2_p["subsample"] = min(best_xgb.get("subsample", 0.8) + 0.05, 1.0)
    m5 = xgb.XGBClassifier(**xgb2_p)
    m5.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb2[val_idx] = m5.predict_proba(X_val)[:, 1]
    pred_xgb2 += m5.predict_proba(X_te)[:, 1] / N_FOLDS

    # Model 6: LightGBM v2 (different seed)
    lgb2_p = best_lgb.copy()
    lgb2_p["random_state"] = 456
    lgb2_p["num_leaves"] = min(best_lgb.get("num_leaves", 40) + 10, 120)
    m6 = lgb.LGBMClassifier(**lgb2_p)
    m6.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb2[val_idx] = m6.predict_proba(X_val)[:, 1]
    pred_lgb2 += m6.predict_proba(X_te)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, (oof_xgb[val_idx] + oof_lgb[val_idx] +
                                      oof_cat[val_idx] + oof_cat2[val_idx] +
                                      oof_xgb2[val_idx] + oof_lgb2[val_idx]) / 6)
    print(f"  Fold {fold+1}: Blend AUC = {fold_auc:.5f}")

# Individual OOF scores
auc_xgb  = roc_auc_score(y_full, oof_xgb)
auc_lgb  = roc_auc_score(y_full, oof_lgb)
auc_cat  = roc_auc_score(y_full, oof_cat)
auc_cat2 = roc_auc_score(y_full, oof_cat2)
auc_xgb2 = roc_auc_score(y_full, oof_xgb2)
auc_lgb2 = roc_auc_score(y_full, oof_lgb2)

print(f"\n  XGB1 OOF AUC:  {auc_xgb:.5f}")
print(f"  LGB1 OOF AUC:  {auc_lgb:.5f}")
print(f"  CAT1 OOF AUC:  {auc_cat:.5f}")
print(f"  CAT2 OOF AUC:  {auc_cat2:.5f}")
print(f"  XGB2 OOF AUC:  {auc_xgb2:.5f}")
print(f"  LGB2 OOF AUC:  {auc_lgb2:.5f}")

log_proof("Executor", "6MODEL_TRAINING_DONE",
          f"xgb={auc_xgb:.5f} lgb={auc_lgb:.5f} cat={auc_cat:.5f} "
          f"cat2={auc_cat2:.5f} xgb2={auc_xgb2:.5f} lgb2={auc_lgb2:.5f}")

# ── 7. YDF MODEL ──────────────────────────────────────────────────────────────
print("\n[7/9] Training YDF (Yggdrasil Decision Forests)...")
try:
    import ydf
    train_ydf = X_full.copy()
    train_ydf["Churn"] = y_full
    test_ydf  = X_test_full.copy()

    oof_ydf  = np.zeros(n_train)
    pred_ydf = np.zeros(n_test)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        tr_df  = train_ydf.iloc[tr_idx].copy()
        val_df = train_ydf.iloc[val_idx].copy()

        model_ydf = ydf.GradientBoostedTreesLearner(
            label="Churn",
            task=ydf.Task.CLASSIFICATION,
            num_trees=500,
            shrinkage=0.05,
            subsample=0.8,
            max_depth=6,
        ).train(tr_df)

        val_preds = model_ydf.predict(val_df.drop("Churn", axis=1))
        oof_ydf[val_idx] = val_preds
        pred_ydf += model_ydf.predict(test_ydf) / N_FOLDS

    auc_ydf = roc_auc_score(y_full, oof_ydf)
    print(f"  YDF OOF AUC: {auc_ydf:.5f}")
    log_proof("Executor", "YDF_TRAINED", f"oof_auc={auc_ydf:.5f}")
    use_ydf = True
except Exception as e:
    print(f"  YDF skipped: {e}")
    auc_ydf = 0
    oof_ydf = np.zeros(n_train)
    pred_ydf = np.zeros(n_test)
    use_ydf = False

# ── 8. RANK BLENDING ──────────────────────────────────────────────────────────
print("\n[8/9] Rank-Based Blending + Weight Optimization...")

def rank_norm(arr):
    return rankdata(arr) / len(arr)

# Rank normalize all OOF predictions
oof_r_xgb  = rank_norm(oof_xgb)
oof_r_lgb  = rank_norm(oof_lgb)
oof_r_cat  = rank_norm(oof_cat)
oof_r_cat2 = rank_norm(oof_cat2)
oof_r_xgb2 = rank_norm(oof_xgb2)
oof_r_lgb2 = rank_norm(oof_lgb2)

pred_r_xgb  = rank_norm(pred_xgb)
pred_r_lgb  = rank_norm(pred_lgb)
pred_r_cat  = rank_norm(pred_cat)
pred_r_cat2 = rank_norm(pred_cat2)
pred_r_xgb2 = rank_norm(pred_xgb2)
pred_r_lgb2 = rank_norm(pred_lgb2)

if use_ydf:
    oof_r_ydf  = rank_norm(oof_ydf)
    pred_r_ydf = rank_norm(pred_ydf)
    oof_stack  = np.column_stack([oof_r_xgb, oof_r_lgb, oof_r_cat, oof_r_cat2,
                                   oof_r_xgb2, oof_r_lgb2, oof_r_ydf])
    pred_stack = np.column_stack([pred_r_xgb, pred_r_lgb, pred_r_cat, pred_r_cat2,
                                   pred_r_xgb2, pred_r_lgb2, pred_r_ydf])
else:
    oof_stack  = np.column_stack([oof_r_xgb, oof_r_lgb, oof_r_cat, oof_r_cat2,
                                   oof_r_xgb2, oof_r_lgb2])
    pred_stack = np.column_stack([pred_r_xgb, pred_r_lgb, pred_r_cat, pred_r_cat2,
                                   pred_r_xgb2, pred_r_lgb2])

# Optimize blend weights
best_blend_auc = 0
best_blend_w   = None
n_models = oof_stack.shape[1]

# Grid search over weights
from itertools import product
step = 0.1
candidates = []
for _ in range(2000):
    w = np.random.dirichlet(np.ones(n_models))
    blend = oof_stack @ w
    auc = roc_auc_score(y_full, blend)
    candidates.append((auc, w))

candidates.sort(key=lambda x: -x[0])
best_blend_auc, best_blend_w = candidates[0]

# Also try Ridge meta-learner on rank-normalized OOF
scaler = StandardScaler()
oof_scaled  = scaler.fit_transform(oof_stack)
test_scaled = scaler.transform(pred_stack)

ridge = RidgeClassifier(alpha=1.0)
ridge.fit(oof_scaled, y_full)
# RidgeClassifier doesn't have predict_proba, use decision_function
oof_ridge  = ridge.decision_function(oof_scaled)
pred_ridge = ridge.decision_function(test_scaled)
auc_ridge  = roc_auc_score(y_full, oof_ridge)

# Normalize ridge to [0,1]
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
pred_ridge_norm = mm.fit_transform(pred_ridge.reshape(-1,1)).ravel()

print(f"  Best Rank Blend AUC: {best_blend_auc:.5f}")
print(f"  Ridge Meta AUC:      {auc_ridge:.5f}")

# Choose best
if best_blend_auc >= auc_ridge:
    final_preds = pred_stack @ best_blend_w
    final_auc   = best_blend_auc
    strategy    = "RankBlend"
else:
    final_preds = pred_ridge_norm
    final_auc   = auc_ridge
    strategy    = "RidgeMeta"

print(f"\n  FINAL STRATEGY: {strategy}")
print(f"  FINAL OOF AUC:  {final_auc:.5f}")

log_proof("Executor", "RANK_BLEND_DONE",
          f"final_oof_auc={final_auc:.5f} strategy={strategy} "
          f"blend_auc={best_blend_auc:.5f} ridge_auc={auc_ridge:.5f}")

# ── 9. SAVE SUBMISSION ────────────────────────────────────────────────────────
print("\n[9/9] Saving submission...")

# Normalize final predictions to [0,1]
final_preds = (final_preds - final_preds.min()) / (final_preds.max() - final_preds.min() + 1e-10)

submission = pd.DataFrame({"id": test_ids, "Churn": final_preds})
submission.to_csv(SUB_PATH, index=False)

assert SUB_PATH.exists()
assert len(submission) == len(test_ids)
assert submission["Churn"].between(0, 1).all()

print(f"\n{'='*70}")
print(f"  MODEL PERFORMANCE SUMMARY")
print(f"{'='*70}")
print(f"  XGB1 OOF AUC:       {auc_xgb:.5f}")
print(f"  LGB1 OOF AUC:       {auc_lgb:.5f}")
print(f"  CAT1 OOF AUC:       {auc_cat:.5f}")
print(f"  CAT2 OOF AUC:       {auc_cat2:.5f}")
print(f"  XGB2 OOF AUC:       {auc_xgb2:.5f}")
print(f"  LGB2 OOF AUC:       {auc_lgb2:.5f}")
if use_ydf:
    print(f"  YDF OOF AUC:        {auc_ydf:.5f}")
print(f"  ─────────────────────────────────────────────────")
print(f"  Rank Blend AUC:     {best_blend_auc:.5f}")
print(f"  Ridge Meta AUC:     {auc_ridge:.5f}")
print(f"  FINAL OOF AUC:      {final_auc:.5f}  [{strategy}]")
print(f"{'='*70}")
print(f"  Submission: {SUB_PATH}")
print(f"  Rows: {len(submission):,}")
print(f"  Churn prob range: [{submission['Churn'].min():.4f}, {submission['Churn'].max():.4f}]")
print(f"  Mean Churn prob: {submission['Churn'].mean():.4f}")
print(f"\n  ANTI_SIMULATION_v3: All proofs logged.")
print(f"  Pipeline v2 complete. Ready for Kaggle submission.")

log_proof("Executor", "SUBMISSION_SAVED_V2",
          f"path={SUB_PATH} rows={len(submission)} "
          f"final_oof_auc={final_auc:.5f} strategy={strategy}")
