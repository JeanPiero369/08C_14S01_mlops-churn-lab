"""Preparación de datos - ejecutable desde CLI o GitHub Actions."""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
os.makedirs(DATA_DIR, exist_ok=True)

RAW_FILENAME = 'telco_customer_churn_mlops.csv'


def load_raw(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, RAW_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Dataset no encontrado: {path}\n'
            f'Asegúrate de que {RAW_FILENAME} esté en {data_dir}/'
        )
    df = pd.read_csv(path)
    print(f'Dataset cargado: {df.shape} — {path}')
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'customer_id':             'customerID',
        'senior_citizen':          'SeniorCitizen',
        'tenure_months':           'tenure',
        'contract_type':           'Contract',
        'internet_service':        'InternetService',
        'monthly_charges':         'MonthlyCharges',
        'support_tickets_last_6m': 'support_tickets',
        'churn':                   'Churn',
    })

    df = df.drop(columns=['customerID'], errors='ignore')

    # Imputar nulos en InternetService
    df['InternetService'] = df['InternetService'].fillna('No')

    # Target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Encoding binario
    df['gender'] = LabelEncoder().fit_transform(df['gender'])

    # One-hot para categóricas restantes
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def main():
    df_raw = load_raw()
    df = preprocess(df_raw.copy())

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_features = ['tenure', 'MonthlyCharges', 'support_tickets']
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))
    pd.Series(X_train.columns.tolist()).to_csv(
        os.path.join(DATA_DIR, 'feature_names.csv'), index=False, header=False
    )

    print(f'Train: {X_train.shape} | Test: {X_test.shape}')
    print('Artefactos guardados en data/')


if __name__ == '__main__':
    main()
