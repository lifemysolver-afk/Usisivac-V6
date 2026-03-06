
# Uvoz potrebnih biblioteka
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# Učitaj podatke
df = pd.read_csv('your_data.csv')

# Odaberite karakteristične podatke i ciljnu promenljivu
X = df.drop('churn', axis=1)
y = df['churn']

# Definujte early stopping criterijume
early_stopping_rounds = 10
eval_metric = 'auc'

# Podelite podatke u trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definujte funkciju za treniranje modela
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Definujte funkciju za procenu modela
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

# Definujte XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Definujte LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Definujte CatBoost model
cb_model = cb.CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=6, random_state=42)

# Definujte pipeline za feature engineering
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Definujte ensemble model
models = [xgb_model, lgb_model, cb_model]
ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cb', cb_model)], voting='soft')

# Trenirajte model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ensemble', ensemble_model)])
pipeline.fit(X_train, y_train)

# Procenite model
y_pred = pipeline.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_pred))

# Definujte funkciju za treniranje modela sa early stopping-om
def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds)
    return model

# Podelite podatke u trening i validacioni skup
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Trenirajte model sa early stopping-om
xgb_model_with_early_stopping = train_model_with_early_stopping(xgb_model, X_train, y_train, X_val, y_val)
lgb_model_with_early_stopping = train_model_with_early_stopping(lgb_model, X_train, y_train, X_val, y_val)
cb_model_with_early_stopping = train_model_with_early_stopping(cb_model, X_train, y_train, X_val, y_val)

# Procenite modele sa early stopping-om
y_pred_xgb = xgb_model_with_early_stopping.predict_proba(X_test)[:, 1]
y_pred_lgb = lgb_model_with_early_stopping.predict_proba(X_test)[:, 1]
y_pred_cb = cb_model_with_early_stopping.predict_proba(X_test)[:, 1]
print("AUC XGB:", roc_auc_score(y_test, y_pred_xgb))
print("AUC LGB:", roc_auc_score(y_test, y_pred_lgb))
print("AUC CB:", roc_auc_score(y_test, y_pred_cb))

# Definujte stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Trenirajte model sa stratified k-fold-om
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Trenirajte model sa early stopping-om
    xgb_model_with_early_stopping = train_model_with_early_stopping(xgb_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    lgb_model_with_early_stopping = train_model_with_early_stopping(lgb_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    cb_model_with_early_stopping = train_model_with_early_stopping(cb_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    
    # Procenite modele sa early stopping-om
    y_pred_xgb = xgb_model_with_early_stopping.predict_proba(X_test)[:, 1]
    y_pred_lgb = lgb_model_with_early_stopping.predict_proba(X_test)[:, 1]
    y_pred_cb = cb_model_with_early_stopping.predict_proba(X_test)[:, 1]
    print("Fold:", fold+1)
    print("AUC XGB:", roc_auc_score(y_test, y_pred_xgb))
    print("AUC LGB:", roc_auc_score(y_test, y_pred_lgb))
    print("AUC CB:", roc_auc_score(y_test, y_pred_cb))
