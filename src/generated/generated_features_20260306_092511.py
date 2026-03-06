
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Učitaj podatke
df = pd.read_csv('customer_churn_dataset.csv')

# Feature 1: Broj dana od poslednje aktivnosti
# Koristan jer pokazuje koliko je korisnik aktivan
df['days_since_last_activity'] = (df['last_activity_date'] - df['account_creation_date']).dt.days

# Feature 2: Broj prijava u poslednjih 30 dana
# Koristan jer pokazuje koliko je korisnik aktivan u poslednje vreme
df['logins_last_30_days'] = df.apply(lambda row: len(row['login_history'][-30:]), axis=1)

# Feature 3: Prosecna ocena korisnika
# Koristan jer pokazuje koliko je korisnik zadovoljan uslugom
df['average_rating'] = df['ratings'].apply(lambda x: np.mean(x))

# Feature 4: Broj prijatelja
# Koristan jer pokazuje koliko je korisnik socijalan
df['number_of_friends'] = df['friends_list'].apply(lambda x: len(x))

# Feature 5:Prosecna vrednost potrošnje
# Koristan jer pokazuje koliko je korisnik vredan za kompaniju
df['average_spend'] = df['purchase_history'].apply(lambda x: np.mean([y['amount'] for y in x]))

# Normalizacija podataka
scaler = StandardScaler()
df[['days_since_last_activity', 'logins_last_30_days', 'average_rating', 'number_of_friends', 'average_spend']] = scaler.fit_transform(df[['days_since_last_activity', 'logins_last_30_days', 'average_rating', 'number_of_friends', 'average_spend']])

# Selekcija najboljih feature-a
X = df.drop(['churn', 'user_id'], axis=1)
y = df['churn']
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Podela podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Kreiranje pipeline-a
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(mutual_info_classif, k=10)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Obuka modela
pipeline.fit(X_train, y_train)
