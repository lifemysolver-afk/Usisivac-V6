
# Uvoz potrebnih biblioteka
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Učitaj podatke
data = pd.read_csv('your_data.csv')

# Izbaci redove sa nedostajućim vrijednostima
data = data.dropna()

# Odredi kategorijke varijable
categorical_cols = data.select_dtypes(include=['object']).columns

# Postavi target varijablu
target = 'churn'

# Podijelipodatke u trening i test skup
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicijalizuj skalere
scaler = StandardScaler()

# Fitiraj skalere samo na trening skupu
X_train[categorical_cols] = X_train[categorical_cols].apply(lambda x: pd.Categorical(x).codes)
X_test[categorical_cols] = X_test[categorical_cols].apply(lambda x: pd.Categorical(x).codes)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definisi modele
xgb_model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=5, n_estimators=100, seed=42)
lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=0.1, max_depth=5, n_estimators=100, seed=42)
cb_model = cb.CatBoostClassifier(loss_function='Logloss', learning_rate=0.1, max_depth=5, n_estimators=100, seed=42)

# Postavi early stopping
early_stopping_rounds = 10

# Trening modela
xgb_model.fit(X_train_scaled, y_train, early_stopping_rounds=early_stopping_rounds)
lgb_model.fit(X_train_scaled, y_train, early_stopping_rounds=early_stopping_rounds)
cb_model.fit(X_train_scaled, y_train, early_stopping_rounds=early_stopping_rounds)

# Evaluacija modela
xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
cb_pred = cb_model.predict_proba(X_test_scaled)[:, 1]

# Izračunaj AUC
xgb_auc = roc_auc_score(y_test, xgb_pred)
lgb_auc = roc_auc_score(y_test, lgb_pred)
cb_auc = roc_auc_score(y_test, cb_pred)

# Ispiši rezultate
print('XGBoost AUC:', xgb_auc)
print('LightGBM AUC:', lgb_auc)
print('CatBoost AUC:', cb_auc)

# Kombiniraj predikcije
ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3

# Izračunaj AUC za ensemble model
ensemble_auc = roc_auc_score(y_test, ensemble_pred)

# Ispiši rezultate
print('Ensemble AUC:', ensemble_auc)
