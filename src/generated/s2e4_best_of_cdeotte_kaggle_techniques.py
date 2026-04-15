# %% [markdown]
# # s2e4: Best of cdeotte's Kaggle Techniques
#
# **Season 2, Episode 4** - Distilled knowledge from 36 notebooks by @cdeotte
#
# This notebook consolidates the winning techniques across:
# - XGBoost feature engineering
# - Neural Network architectures
# - Transformer models (DeBERTa, Gemma, ModernBERT)
# - Ensemble methods
# - GPU acceleration patterns
#
# ---
#
# ## Table of Contents
# 1. [XGBoost Best Practices](#xgboost)
# 2. [Neural Network Patterns](#nn)
# 3. [Transformer Fine-tuning](#transformers)
# 4. [Ensemble Techniques](#ensemble)
# 5. [GPU Optimization](#gpu)
#
# ---

# %% [markdown]
# ## 1. XGBoost Best Practices <a id='xgboost'></a>

# %% [code]

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# === XGBoost Parameters (proven for tabular competitions) ===
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
}

# === K-Fold Training Pattern ===
def train_xgb_kfold(X, y, params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof[val_idx] = model.predict(dval)
        models.append(model)

    return oof, models


# %% [markdown]
# ### Feature Engineering Patterns

# %% [code]

# === Cross-Feature Creation ===
def create_features(df):
    # Interaction features
    df['feat1_x_feat2'] = df['feat1'] * df['feat2']
    df['feat1_div_feat2'] = df['feat1'] / (df['feat2'] + 1e-5)

    # Statistical aggregations by group
    for col in ['cat1', 'cat2']:
        agg = df.groupby(col)['target_col'].agg(['mean', 'std', 'count'])
        df[f'{col}_mean'] = df[col].map(agg['mean'])
        df[f'{col}_std'] = df[col].map(agg['std'])

    # Polynomial features
    df['feat1_sq'] = df['feat1'] ** 2
    df['feat1_sqrt'] = np.sqrt(np.abs(df['feat1']))

    return df

# === Target Encoding with Regularization ===
from sklearn.model_selection import KFold

def target_encode(train_df, test_df, cols, target, n_splits=5):
    for col in cols:
        train_df[f'{col}_te'] = 0.0

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(train_df):
            means = train_df.iloc[train_idx].groupby(col)[target].mean()
            train_df.loc[train_df.index[val_idx], f'{col}_te'] = \
                train_df.loc[train_df.index[val_idx], col].map(means)

        global_mean = train_df[target].mean()
        train_df[f'{col}_te'].fillna(global_mean, inplace=True)

        test_means = train_df.groupby(col)[target].mean()
        test_df[f'{col}_te'] = test_df[col].map(test_means).fillna(global_mean)

    return train_df, test_df


# %% [markdown]
# ### Boosting Over Residuals

# %% [code]

# === Boosting Over Residuals ===
# When you have an optimal Bayesian solution (baseline),
# train XGBoost on the RESIDUAL = target - baseline

def train_over_residuals(X, y, baseline_predictions, params):
    # Calculate residuals
    residuals = y - baseline_predictions

    # Train XGB on residuals
    dtrain = xgb.DMatrix(X, label=residuals)
    model = xgb.train(params, dtrain, num_boost_round=500)

    # Final prediction = baseline + XGB residual correction
    xgb_correction = model.predict(dtrain)
    final_pred = baseline_predictions + xgb_correction

    return final_pred, model


# %% [markdown]
# ## 2. Neural Network Patterns <a id='nn'></a>

# %% [code]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === MLP Architecture ===
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)

# === Training Loop ===
def train_nn(model, train_loader, val_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for batch in val_loader:
                x, y = batch
                pred = torch.sigmoid(model(x))
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

    return model


# %% [markdown]
# ## 3. Transformer Fine-tuning <a id='transformers'></a>

# %% [code]

# === DeBERTa for Text Classification ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_name = "microsoft/deberta-v3-xsmall"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    fp16=True,
    dataloader_num_workers=4,
)


# %% [markdown]
# ## 4. Ensemble Techniques <a id='ensemble'></a>

# %% [code]

# === XGB + NN Ensemble ===
def ensemble_predictions(xgb_preds, nn_preds, weights=[0.5, 0.5]):
    """Blend XGBoost and Neural Network predictions"""
    return weights[0] * xgb_preds + weights[1] * nn_preds

# === Hill Climbing for Optimal Weights ===
def hill_climb_weights(oof_xgb, oof_nn, y_true, n_iter=1000):
    best_score = 0
    best_weights = [0.5, 0.5]

    for _ in range(n_iter):
        w = np.random.dirichlet([1, 1])
        pred = w[0] * oof_xgb + w[1] * oof_nn
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_true, pred)

        if score > best_score:
            best_score = score
            best_weights = w.tolist()

    return best_weights, best_score

# === K-Fold Ensemble Pattern ===
def create_kfold_ensemble(X, y, model_fn, n_splits=5):
    """Train multiple models across K folds for robust ensemble"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_fn()
        all_preds.append((fold, model))

    return all_preds


# %% [markdown]
# ## 5. GPU Optimization <a id='gpu'></a>

# %% [code]

# === RAPIDS for GPU Acceleration ===
try:
    import cudf
    import cuml
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False
    print("RAPIDS not available, using CPU")

if HAS_RAPIDS:
    from cuml.linear_model import LogisticRegression
    from cuml.ensemble import RandomForestClassifier

    # Convert pandas to cuDF for GPU acceleration
    train_gdf = cudf.DataFrame.from_pandas(train_df)
    test_gdf = cudf.DataFrame.from_pandas(test_df)

# === GPU XGBoost ===
import xgboost as xgb

dtrain = xgb.DMatrix(train_gdf, label=y_train)
dtest = xgb.DMatrix(test_gdf)

xgb_params_gpu = {
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'objective': 'binary:logistic',
}

# === GPU KNN ===
if HAS_RAPIDS:
    from cuml.neighbors import KNeighborsClassifier

    knn_model = KNeighborsClassifier(n_neighbors=100, algorithm='brute')
    knn_model.fit(train_gdf, y_train)
    predictions = knn_model.predict(test_gdf)


# %% [markdown]
# ---
#
# ## Key Takeaways
#
# 1. **XGBoost + Original Data**: Using original competition data dramatically improves CV
# 2. **Feature Engineering**: Cross-features, target encoding, polynomial features
# 3. **Ensemble Diversity**: XGB + NN + Transformers gives best results
# 4. **GPU Acceleration**: RAPIDS (cuML) can 10x speed up for tabular data
# 5. **Residual Boosting**: When you have a good baseline, train on residuals
#
# ---
#
# *Generated: 2026-04-15 06:49*
# *Source: 36 notebooks by @cdeotte from Kaggle Playground competitions*
