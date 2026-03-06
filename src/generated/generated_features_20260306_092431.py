
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def generate_features(df):
    # 1. Koliko dana je potrebno da se desi churn
    # Ovo je korisno jer nam daje informaciju o tome koliko dugo customer nije imao nikakvih interakcija sa kompanijom
    df['days_before_churn'] = df['churn_date'] - df['last_interaction_date']
    
    # 2. Broj koriscenih usluga
    # Ovo je korisno jer nam daje informaciju o tome koliko je customer koristio usluge kompanije
    df['num_services_used'] = df[['service1', 'service2', 'service3']].sum(axis=1)
    
    # 3. Ukupan broj poziva
    # Ovo je korisno jer nam daje informaciju o tome koliko je customer bio u kontaktu sa kompanijom
    df['total_calls'] = df['incoming_calls'] + df['outgoing_calls']
    
    # 4. Prosecna ocena kompanije
    # Ovo je korisno jer nam daje informaciju o tome koliko je customer zadovoljan uslugama kompanije
    df['avg_rating'] = df[['rating1', 'rating2', 'rating3']].mean(axis=1)
    
    # 5. Broj mjeseci od prvog poziva do chyna
    # Ovo je korisno jer nam daje informaciju o tome koliko dugo customer koristi usluge kompanije
    df['months_from_first_call_to_churn'] = (df['churn_date'] - df['first_call_date']).dt.days / 30
    
    # 6. Da li je customer koristio specijalnu ponudu
    # Ovo je korisno jer nam daje informaciju o tome da li je customer koristio specijalnu ponudu kompanije
    df['used_special_offer'] = np.where(df['special_offer'] == 1, 1, 0)
    
    return df

def feature_engineering_pipeline(df):
    df = generate_features(df)
    
    # Standardizacija
    scaler = StandardScaler()
    df[['num_services_used', 'total_calls', 'avg_rating']] = scaler.fit_transform(df[['num_services_used', 'total_calls', 'avg_rating']])
    
    # PCA
    pca = PCA(n_components=0.9)
    df_pca = pca.fit_transform(df[['num_services_used', 'total_calls', 'avg_rating']])
    df_pca = pd.DataFrame(df_pca, columns=['pca1', 'pca2', 'pca3'])
    df = pd.concat([df, df_pca], axis=1)
    
    # Selekcija najboljih feature-a
    selector = SelectKBest(mutual_info_classif, k=10)
    df_selected = selector.fit_transform(df, df['churn'])
    df_selected = pd.DataFrame(df_selected, columns=[f'feature_{i}' for i in range(10)])
    df = pd.concat([df, df_selected], axis=1)
    
    return df
