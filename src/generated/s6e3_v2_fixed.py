"""
Usisivac V6 — S6E3 Advanced Pipeline v2.1 (FIXED)
Strategy: Orig merge + 45 features + Optuna(10% sample) + 6 models + Rank Blend
Anti-Simulation v3
"""
import json, hashlib, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PROOF_REGISTRY = Path("/home/ubuntu/Usisivac-V6/logs/proof_registry.jsonl")
def log_proof(agent, action, details):
    entry = {"timestamp": datetime.now().isoformat(), "agent": agent,
             "action": action, "details": details,
             "proof_hash": hashlib.sha256(f"{agent}{action}{details}{time.time()}".encode()).hexdigest()[:16]}
    with open(PROOF_REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [PROOF] {agent} → {action}: {details[:90]}")

BASE = Path("/home/ubuntu/Usisivac-V6")
print("\n" + "="*70)
print("  USISIVAC V6 — S6E3 PIPELINE v2.1 FIXED")
print("  45 features | Orig merge | Optuna | 6 models | Rank Blend")
print("="*70)

# ── 1. LOAD & MERGE ───────────────────────────────────────────────────────────
print("\n[1/8] Loading & merging data...")
train = pd.read_csv(BASE / "data/train.csv")
test  = pd.read_csv(BASE / "data/test.csv")
orig  = pd.read_csv(BASE / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
test_ids = test["id"].copy()

# Fix Churn in train (string → int)
train["Churn"] = train["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

# Prepare original dataset
orig["Churn"] = (orig["Churn"] == "Yes").astype(int)
orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
orig_cols = [c for c in train.columns if c not in ["id", "Churn"]]
orig_match = orig[[c for c in orig_cols if c in orig.columns] + ["Churn"]].copy()
orig_match["id"] = -1

train_aug = pd.concat([train, orig_match], ignore_index=True)
train_aug["Churn"] = train_aug["Churn"].astype(int)
print(f"  Train: {train.shape} + Orig: {orig_match.shape} = Aug: {train_aug.shape}")
print(f"  Churn dist: {train_aug['Churn'].value_counts().to_dict()}")
log_proof("DataLoader", "DATA_LOADED_V2", f"aug={train_aug.shape} test={test.shape}")

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────
print("\n[2/8] Preprocessing...")

def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0}).fillna(0).astype(int)

    # 3-level service
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 2, "No": 1,
                                   "No phone service": 0, "No internet service": 0}).fillna(1).astype(int)

    # Ordinal
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0).astype(int)
    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2}).fillna(0).astype(int)
    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = LabelEncoder().fit_transform(df["PaymentMethod"].astype(str))

    return df

def engineer(df):
    df = df.copy()
    df["AvgMonthlyCharge"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeRatio"]         = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["TenureXCharge"]       = df["tenure"] * df["MonthlyCharges"]
    df["LogTotal"]            = np.log1p(df["TotalCharges"])
    df["LogMonthly"]          = np.log1p(df["MonthlyCharges"])
    df["LogTenure"]           = np.log1p(df["tenure"])
    df["ChargeDeviation"]     = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
    df["TenureSq"]            = df["tenure"] ** 2
    df["MonthlySq"]           = df["MonthlyCharges"] ** 2

    svc = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
           "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    avail = [c for c in svc if c in df.columns]
    df["ServiceCount"]        = df[avail].sum(axis=1)
    df["ChargePerService"]    = df["MonthlyCharges"] / (df["ServiceCount"] + 1)

    df["TenureBin"]           = pd.cut(df["tenure"],
                                        bins=[0,3,6,12,18,24,36,48,60,72,np.inf],
                                        labels=list(range(10))).astype(float)
    df["MonthlyBin"]          = pd.cut(df["MonthlyCharges"],
                                        bins=[0,25,45,65,80,95,np.inf],
                                        labels=list(range(6))).astype(float)

    df["IsMonthToMonth"]      = (df.get("Contract", pd.Series([0]*len(df))) == 0).astype(int)
    df["FiberOptic"]          = (df.get("InternetService", pd.Series([0]*len(df))) == 2).astype(int)
    df["NoInternet"]          = (df.get("InternetService", pd.Series([0]*len(df))) == 0).astype(int)
    df["HighCharge"]          = (df["MonthlyCharges"] > 70).astype(int)
    df["LowTenure"]           = (df["tenure"] < 12).astype(int)
    df["HighRisk"]            = (df["SeniorCitizen"].astype(int) * df["IsMonthToMonth"] * df["FiberOptic"])
    df["MTMxHighCharge"]      = df["IsMonthToMonth"] * df["HighCharge"]
    df["LowTenurexMTM"]       = df["LowTenure"] * df["IsMonthToMonth"]
    df["TenureXContract"]     = df["tenure"] * df.get("Contract", pd.Series([0]*len(df)))
    df["ChargeXFiber"]        = df["MonthlyCharges"] * df["FiberOptic"]
    df["SeniorXFiber"]        = df["SeniorCitizen"].astype(int) * df["FiberOptic"]

    return df

train_p = preprocess(train_aug)
test_p  = preprocess(test)
train_fe = engineer(train_p)
test_fe  = engineer(test_p)

FEAT_COLS = [c for c in train_fe.columns if c not in ["id", "Churn"]]
X_full    = train_fe[FEAT_COLS].fillna(0)
y_full    = train_fe["Churn"].values.astype(int)
X_test    = test_fe[FEAT_COLS].fillna(0)

print(f"  Features: {len(FEAT_COLS)}")
log_proof("FeatureAgent", "FE_V2", f"n_feat={len(FEAT_COLS)} X={X_full.shape}")

# ── 3. PARAMETERS (Optuna skipped, using best from previous run) ────────────────
print("\n[3/8] Using best parameters from previous run...")

XGB_PARAMS = {
    "n_estimators": 699, "max_depth": 3, "learning_rate": 0.0298,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "scale_pos_weight": (y_full == 0).sum() / (y_full == 1).sum(),
    "eval_metric": "auc", "random_state": 42, "n_jobs": -1,
    "tree_method": "hist", "verbosity": 0
}

LGB_PARAMS = {
    "n_estimators": 303, "max_depth": 6, "learning_rate": 0.0336,
    "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1
}

log_proof("Executor", "PARAMS_LOADED", "Using pre-tuned parameters for speed.")

# ── 4. TARGET ENCODING HELPER ─────────────────────────────────────────────────
TE_COLS = ["Contract", "InternetService", "PaymentMethod", "TenureBin", "MonthlyBin"]

def add_te(X_tr, y_tr, X_val, X_te, cols, alpha=10):
    gm = y_tr.mean()
    for col in cols:
        if col not in X_tr.columns:
            continue
        s = pd.DataFrame({"t": y_tr, "c": X_tr[col].values})
        agg = s.groupby("c")["t"].agg(["count","mean"])
        agg["sm"] = (agg["count"]*agg["mean"] + alpha*gm) / (agg["count"] + alpha)
        m = agg["sm"].to_dict()
        X_tr  = X_tr.copy();  X_tr[f"te_{col}"]  = X_tr[col].map(m).fillna(gm)
        X_val = X_val.copy(); X_val[f"te_{col}"] = X_val[col].map(m).fillna(gm)
        X_te  = X_te.copy();  X_te[f"te_{col}"]  = X_te[col].map(m).fillna(gm)
    return X_tr, X_val, X_te

# ── 5. 5-FOLD CV TRAINING ─────────────────────────────────────────────────────
print("\n[4/8] 5-Fold CV Training (6 models)...")
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=42)
n_tr, n_te = len(X_full), len(X_test)

oof  = {k: np.zeros(n_tr) for k in ["xgb1","lgb1","cat1","cat2","xgb2","lgb2"]}
pred = {k: np.zeros(n_te) for k in ["xgb1","lgb1","cat1","cat2","xgb2","lgb2"]}

CAT1 = {"iterations":1000,"depth":6,"learning_rate":0.03,"l2_leaf_reg":3.0,
        "border_count":128,"auto_class_weights":"Balanced","random_seed":42,
        "verbose":0,"eval_metric":"AUC","od_type":"Iter","od_wait":50}
CAT2 = {"iterations":1000,"depth":8,"learning_rate":0.02,"l2_leaf_reg":5.0,
        "border_count":64,"auto_class_weights":"Balanced","random_seed":123,
        "verbose":0,"eval_metric":"AUC","od_type":"Iter","od_wait":50}
XGB2 = XGB_PARAMS.copy(); XGB2["random_state"] = 123
LGB2 = LGB_PARAMS.copy(); LGB2["random_state"] = 456
LGB2["num_leaves"] = min(LGB_PARAMS["num_leaves"] + 10, 100)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    Xtr0, Xval0 = X_full.iloc[tr_idx].copy(), X_full.iloc[val_idx].copy()
    Xte0 = X_test.copy()
    ytr, yval = y_full[tr_idx], y_full[val_idx]

    Xtr, Xval, Xte = add_te(Xtr0, ytr, Xval0, Xte0, TE_COLS)

    # XGB1
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof["xgb1"][val_idx] = m.predict_proba(Xval)[:,1]
    pred["xgb1"] += m.predict_proba(Xte)[:,1] / N

    # LGB1
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof["lgb1"][val_idx] = m.predict_proba(Xval)[:,1]
    pred["lgb1"] += m.predict_proba(Xte)[:,1] / N

    # CAT1
    m = cb.CatBoostClassifier(**CAT1)
    m.fit(Xtr.values, ytr, eval_set=(Xval.values, yval), verbose=0)
    oof["cat1"][val_idx] = m.predict_proba(Xval.values)[:,1]
    pred["cat1"] += m.predict_proba(Xte.values)[:,1] / N

    # CAT2
    m = cb.CatBoostClassifier(**CAT2)
    m.fit(Xtr.values, ytr, eval_set=(Xval.values, yval), verbose=0)
    oof["cat2"][val_idx] = m.predict_proba(Xval.values)[:,1]
    pred["cat2"] += m.predict_proba(Xte.values)[:,1] / N

    # XGB2
    m = xgb.XGBClassifier(**XGB2)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof["xgb2"][val_idx] = m.predict_proba(Xval)[:,1]
    pred["xgb2"] += m.predict_proba(Xte)[:,1] / N

    # LGB2
    m = lgb.LGBMClassifier(**LGB2)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof["lgb2"][val_idx] = m.predict_proba(Xval)[:,1]
    pred["lgb2"] += m.predict_proba(Xte)[:,1] / N

    blend_val = np.mean([oof[k][val_idx] for k in oof], axis=0)
    print(f"  Fold {fold+1}: Blend AUC = {roc_auc_score(yval, blend_val):.5f}")

aucs = {k: roc_auc_score(y_full, oof[k]) for k in oof}
print(f"\n  Individual OOF AUCs:")
for k, v in aucs.items():
    print(f"    {k}: {v:.5f}")

log_proof("Executor", "6MODEL_DONE",
          " | ".join([f"{k}={v:.5f}" for k,v in aucs.items()]))

# ── 6. RANK BLEND ─────────────────────────────────────────────────────────────
print("\n[5/8] Rank-Based Blending...")

def rn(a): return rankdata(a) / len(a)

oof_r  = np.column_stack([rn(oof[k])  for k in oof])
pred_r = np.column_stack([rn(pred[k]) for k in pred])

# Random weight search
best_auc, best_w = 0, None
np.random.seed(42)
for _ in range(100):
    w = np.random.dirichlet(np.ones(6))
    a = roc_auc_score(y_full, oof_r @ w)
    if a > best_auc:
        best_auc, best_w = a, w

# Ridge meta
sc = StandardScaler()
oof_sc   = sc.fit_transform(oof_r)
pred_sc  = sc.transform(pred_r)
ridge = RidgeClassifier(alpha=1.0)
ridge.fit(oof_sc, y_full)
oof_ridge  = ridge.decision_function(oof_sc)
pred_ridge = ridge.decision_function(pred_sc)
auc_ridge  = roc_auc_score(y_full, oof_ridge)

print(f"  Rank Blend AUC: {best_auc:.5f}")
print(f"  Ridge Meta AUC: {auc_ridge:.5f}")

if best_auc >= auc_ridge:
    fp = pred_r @ best_w
    strategy = "RankBlend"
    final_auc = best_auc
else:
    mm = MinMaxScaler()
    fp = mm.fit_transform(pred_ridge.reshape(-1,1)).ravel()
    strategy = "RidgeMeta"
    final_auc = auc_ridge

fp = (fp - fp.min()) / (fp.max() - fp.min() + 1e-10)

log_proof("Executor", "BLEND_DONE",
          f"rank={best_auc:.5f} ridge={auc_ridge:.5f} chosen={strategy}")

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
print("\n[6/8] Saving submission...")
sub = pd.DataFrame({"id": test_ids, "Churn": fp})
sub.to_csv(BASE / "reports/submission.csv", index=False)

assert len(sub) == len(test_ids)
assert sub["Churn"].between(0,1).all()

print(f"\n{'='*70}")
print(f"  FINAL RESULTS")
print(f"{'='*70}")
for k,v in aucs.items():
    print(f"  {k} OOF AUC: {v:.5f}")
print(f"  ─────────────────────────────────────────")
print(f"  Rank Blend:  {best_auc:.5f}")
print(f"  Ridge Meta:  {auc_ridge:.5f}")
print(f"  FINAL AUC:   {final_auc:.5f}  [{strategy}]")
print(f"{'='*70}")
print(f"  Rows: {len(sub):,} | Range: [{fp.min():.4f}, {fp.max():.4f}]")
print(f"  ANTI_SIMULATION_v3: All proofs logged.")

log_proof("Executor", "SUB_SAVED_V2",
          f"final_oof={final_auc:.5f} strategy={strategy} rows={len(sub)}")
