
# Uvozi potrebne biblioteke
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OptimalBinning
import numpy as np
import pandas as pd

# Učitaj podatke
df = pd.read_csv('data.csv')

# Odaberite relevantne kolone
X = df.drop(['target'], axis=1)
y = df['target']

# Kreirajte funkciju za feature engineering
def feature_engineering(X):
    # Izvrši target encoding
    X['target_encoded'] = X['feature'].map(y.value_counts())
    
    # Dodajte OptimalBinning
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    binning = OptimalBinning()
    X[numeric_features] = binning.fit_transform(X[numeric_features])
    
    return X

# Podijelite podatke u trening i validacijski skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kreirajte funkciju za treniranje modela
def train_model(X_train, y_train, model):
    # Dodajte early stopping
    from sklearn.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Dodajte Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Trenirajte modele
    models = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Kreirajte pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Trenirajte model
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Dodajte early stopping
        pipeline.fit(X_train_fold, y_train_fold, callbacks=[early_stopping])
        
        # Dodajte model u listu
        models.append(pipeline)
    
    return models

# Kreirajte modele
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
lgbm_model = LGBMClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
catboost_model = CatBoostClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)

# Trenirajte modele
xgb_models = train_model(X_train, y_train, xgb_model)
lgbm_models = train_model(X_train, y_train, lgbm_model)
catboost_models = train_model(X_train, y_train, catboost_model)

# Kreirajte funkciju za predviđanje
def predict(models, X_test):
    predictions = []
    for model in models:
        prediction = model.predict_proba(X_test)[:, 1]
        predictions.append(prediction)
    
    # Kreirajte prosječnu predviđanje
    average_prediction = np.mean(predictions, axis=0)
    
    return average_prediction

# Predviđajte target varijablu
xgb_predictions = predict(xgb_models, X_test)
lgbm_predictions = predict(lgbm_models, X_test)
catboost_predictions = predict(catboost_models, X_test)

# Kreirajte funkciju za izračunavanje AUC metrike
def calculate_auc(y_test, predictions):
    auc = roc_auc_score(y_test, predictions)
    return auc

# Izračunajte AUC metriku
xgb_auc = calculate_auc(y_test, xgb_predictions)
lgbm_auc = calculate_auc(y_test, lgbm_predictions)
catboost_auc = calculate_auc(y_test, catboost_predictions)

# Ispišite AUC metrike
print(f'XGBoost AUC: {xgb_auc:.4f}')
print(f'LightGBM AUC: {lgbm_auc:.4f}')
print(f'CatBoost AUC: {catboost_auc:.4f}')

# Na kraju, kreirajte_blend model koji će kombinovati predviđanja svih modela
blend_predictions = (xgb_predictions + lgbm_predictions + catboost_predictions) / 3
blend_auc = calculate_auc(y_test, blend_predictions)
print(f'Blend AUC: {blend_auc:.4f}')
