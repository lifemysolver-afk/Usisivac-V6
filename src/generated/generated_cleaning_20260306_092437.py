
# Uvoz potrebnih biblioteka
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy import stats

# Učitaj podatke
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Greska prilikom učitavanja podataka: {e}")

# Detektuj i loguj missing values
def detect_missing_values(data):
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print("Detektovani missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("Nema missing values")

# Detektuj outliere
def detect_outliers_z_score(data, column):
    z scores = np.abs(stats.zscore(data[column]))
    return z scores > 3

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

# Primjeni strategije za čišćenje
def impute_missing_values(data, column):
    mean_value = data[column].mean()
    data[column] = data[column].fillna(mean_value)

def remove_outliers(data, column, outliers):
    data = data[~outliers]
    return data

def cap_values(data, column, lower_bound, upper_bound):
    data[column] = np.clip(data[column], lower_bound, upper_bound)
    return data

# Normaliziraj numeričke kolone
def normalize_numerical_columns(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Enkodiraj kategoričke kolone
def encode_categorical_columns(data, columns):
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(data[columns])
    encoded_columns = [f"{column}_{category}" for column in columns for category in encoder.categories_[list(columns).index(column)]]
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(columns, axis=1)
    return data

# Čuvaj čist dataset
def save_clean_data(data, path):
    try:
        data.to_csv(path, index=False)
    except Exception as e:
        print(f"Greska prilikom čuvanja čistog dataset: {e}")

# Generiši cleaning report
def generate_cleaning_report(data_before, data_after):
    report = f"Broj redova prije čišćenja: {data_before.shape[0]}\n"
    report += f"Broj redova nakon čišćenja: {data_after.shape[0]}\n"
    report += f"Broj kolona prije čišćenja: {data_before.shape[1]}\n"
    report += f"Broj kolona nakon čišćenja: {data_after.shape[1]}\n"
    return report

# Glavna funkcija
def main():
    path = "data/input.csv"
    data = load_data(path)
    print("Originalni dataset:")
    print(data.head())

    detect_missing_values(data)

    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    print("Numeričke kolone:")
    print(numerical_columns)

    categorical_columns = data.select_dtypes(include=['object']).columns
    print("Kategoričke kolone:")
    print(categorical_columns)

    for column in numerical_columns:
        outliers_z_score = detect_outliers_z_score(data, column)
        outliers_iqr = detect_outliers_iqr(data, column)
        if outliers_z_score.any() or outliers_iqr.any():
            print(f"Detektovani outlieri u koloni {column}:")
            print(f"Z-score: {outliers_z_score.sum()}")
            print(f"IQR: {outliers_iqr.sum()}")
            data = remove_outliers(data, column, outliers_z_score | outliers_iqr)

        if data[column].isnull().any():
            impute_missing_values(data, column)

    data = normalize_numerical_columns(data, numerical_columns)
    data = encode_categorical_columns(data, categorical_columns)

    save_clean_data(data, "data/clean_data.csv")

    report = generate_cleaning_report(data_before=data, data_after=data)
    print("Izvještaj o čišćenju:")
    print(report)

if __name__ == "__main__":
    main()
