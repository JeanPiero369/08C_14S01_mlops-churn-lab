"""Registra el mejor modelo en MLflow Model Registry y lo mueve a Production."""
import json
import os
import warnings
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MLFLOW_URI = os.path.join(os.path.dirname(__file__), '../mlruns')
REGISTERED_MODEL_NAME = 'churn-predictor'
PRODUCTION_F1_THRESHOLD = 0.50

mlflow.set_tracking_uri(f'file://{os.path.abspath(MLFLOW_URI)}')
client = MlflowClient()


def main():
    with open(os.path.join(DATA_DIR, 'best_model_meta.json')) as f:
        meta = json.load(f)

    best_run_id = meta['best_run_id']
    model_uri = f'runs:/{best_run_id}/model'

    print(f'Registrando: {model_uri}')
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f'Versión {mv.version} registrada')

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=mv.version,
        stage='Staging', archive_existing_versions=False
    )

    # Validar en staging
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

    staging_model = mlflow.sklearn.load_model(f'models:/{REGISTERED_MODEL_NAME}/Staging')
    y_pred = staging_model.predict(X_test)
    staging_f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f'Staging F1: {staging_f1:.4f} | Threshold: {PRODUCTION_F1_THRESHOLD}')

    if staging_f1 >= PRODUCTION_F1_THRESHOLD:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=mv.version,
            stage='Production', archive_existing_versions=True
        )
        prod_model = mlflow.sklearn.load_model(f'models:/{REGISTERED_MODEL_NAME}/Production')
        joblib.dump(prod_model, os.path.join(DATA_DIR, 'production_model.pkl'))

        prod_info = {
            'model_name': REGISTERED_MODEL_NAME,
            'version': str(mv.version),
            'stage': 'Production',
            'run_id': best_run_id,
            'model_type': meta['best_model'],
            'model_uri': f'models:/{REGISTERED_MODEL_NAME}/Production',
        }
        with open(os.path.join(DATA_DIR, 'production_model_info.json'), 'w') as f:
            json.dump(prod_info, f, indent=2)

        print(f'Versión {mv.version} promovida a Production')
    else:
        print(f'Versión NO promovida. F1 insuficiente.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
