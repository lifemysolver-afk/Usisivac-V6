
# Uvoz biblioteka
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Učitaj podatke
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Došlo je do greške pri učitavanju podataka: {e}")

# Detektuj i loguj missing values
def detect_missing_values(data):
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Detektovani missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("Nema missing values.")

# Detektuj outliere
def detect_outliers(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    outliers = {}
    for column in numerical_columns:
        mean = data[column].mean()
        std_dev = data[column].std()
        z_scores = np.abs((data[column] - mean) / std_dev)
        iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
        lower_bound = data[column].quantile(0.25) - 1.5 * iqr
        upper_bound = data[column].quantile(0.75) + 1.5 * iqr
        z_outliers = data[z_scores > 3][column]
        iqr_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
        outliers[column] = {
            'z_score': z_outliers,
            'iqr': iqr_outliers
        }
    return outliers

# Primjeni strategije za čišćenje
def clean_data(data):
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)])

    cleaned_data = preprocessor.fit_transform(data)
    return cleaned_data

# Čuvaj čist dataset
def save_cleaned_data(data, path):
    try:
        pd.DataFrame(data).to_csv(path, index=False)
        print("Čist dataset uspješno spremljen.")
    except Exception as e:
        print(f"Došlo je do greške pri spremanju čistog dataset: {e}")

# Generiši cleaning report
def generate_cleaning_report(data, outliers):
    report = "Cleaning Report\n"
    report += "----------------\n"
    report += "Detektovani missing values:\n"
    missing_values = data.isnull().sum()
    report += missing_values.to_string() + "\n\n"
    report += "Detektovani outlieri:\n"
    for column, outlier in outliers.items():
        report += f"{column}:\n"
        report += f"Z-score outlieri: {outlier['z_score']}\n"
        report += f"IQR outlieri: {outlier['iqr']}\n\n"
    return report

# Glavna funkcija
def main():
    path = 'data/input.csv'
    data = load_data(path)
    detect_missing_values(data)
    outliers = detect_outliers(data)
    cleaned_data = clean_data(data)
    save_cleaned_data(cleaned_data, 'data/cleaned_data.csv')
    report = generate_cleaning_report(data, outliers)
    print(report)

if __name__ == "__main__":
    main()
