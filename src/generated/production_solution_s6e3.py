import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

# Putanje do podataka
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'reports/submission.csv'

def run_production_pipeline():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Target i ID
    TARGET = 'Churn'
    ID = 'id'
    
    # Encode target
    le_target = LabelEncoder()
    train[TARGET] = le_target.fit_transform(train[TARGET].astype(str))
    print(f"Target classes: {le_target.classes_}")
    
    # Osnovna priprema
    X = train.drop([TARGET, ID], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID], axis=1)
    
    # Label Encoding za kategorijalne kolone
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Cross-validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print(f"Starting {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}...")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            tree_method='hist',
            early_stopping_rounds=50
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # 2. LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            verbose=-1
        )
        model_lgb.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)], 
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
        
        # 3. CatBoost
        model_cb = cb.CatBoostClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42, 
            verbose=False,
            early_stopping_rounds=50
        )
        model_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        # Ensemble (Simple Average)
        fold_preds = (model_xgb.predict_proba(X_val)[:, 1] + 
                      model_lgb.predict_proba(X_val)[:, 1] + 
                      model_cb.predict_proba(X_val)[:, 1]) / 3
        
        oof_preds[val_idx] = fold_preds
        
        test_preds += (model_xgb.predict_proba(X_test)[:, 1] + 
                       model_lgb.predict_proba(X_test)[:, 1] + 
                       model_cb.predict_proba(X_test)[:, 1]) / (3 * n_splits)
        
        fold_auc = roc_auc_score(y_val, fold_preds)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {overall_auc:.5f}")
    
    # Generisanje submission fajla
    print(f"Saving submission to {SUBMISSION_PATH}...")
    os.makedirs('reports', exist_ok=True)
    submission = pd.DataFrame({ID: test[ID], TARGET: test_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    run_production_pipeline()
