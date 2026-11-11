"""Entrenamiento de modelos con MLflow tracking. Entrena los 3 modelos y guarda el mejor por F1."""
import argparse
import json
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

DATA_DIR  = os.path.join(os.path.dirname(__file__), '../data')
MLFLOW_URI = os.path.join(os.path.dirname(__file__), '../mlruns')

ALL_MODELS = ['logistic', 'random_forest', 'xgboost']


def get_model(model_type: str, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 2)

    configs = {
        'logistic': (
            LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', solver='lbfgs', random_state=42),
            {'C': 1.0, 'max_iter': 500, 'class_weight': 'balanced'}
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
            {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'}
        ),
        'xgboost': (
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                          random_state=42, eval_metric='logloss', verbosity=0),
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'scale_pos_weight': spw}
        ),
    }
    return configs[model_type]


def train_one(model_type, X_train, X_test, y_train, y_test, experiment):
    model, params = get_model(model_type, y_train)

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.set_tags({'model_type': model_type, 'dataset': 'telco_churn', 'task': 'binary_classification'})
        mlflow.log_params(params)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model', input_example=X_train.head(3))

    return {'model_type': model_type, 'run_id': run.info.run_id, 'metrics': metrics, 'model_obj': model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all', choices=['logistic', 'random_forest', 'xgboost', 'all'],
                        help='Modelo a entrenar. "all" entrena los 3 y elige el mejor por F1.')
    parser.add_argument('--experiment', default='churn-prediction-fintech')
    args = parser.parse_args()

    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

    mlflow.set_tracking_uri(f'file://{os.path.abspath(MLFLOW_URI)}')
    mlflow.set_experiment(args.experiment)

    to_train = ALL_MODELS if args.model == 'all' else [args.model]
    results  = []

    for model_type in to_train:
        print(f'Entrenando {model_type}...')
        result = train_one(model_type, X_train, X_test, y_train, y_test, args.experiment)
        results.append(result)
        print(f'  F1={result["metrics"]["f1"]:.4f}  ROC={result["metrics"]["roc_auc"]:.4f}')

    # Seleccionar mejor por F1
    best = max(results, key=lambda r: r['metrics']['f1'])

    print(f'\nMejor modelo: {best["model_type"]} (F1={best["metrics"]["f1"]:.4f})')

    meta = {
        'best_model':  best['model_type'],
        'best_run_id': best['run_id'],
        'metrics':     best['metrics'],
        'experiment':  args.experiment,
        'all_results': [{'model': r['model_type'], 'f1': r['metrics']['f1'], 'roc_auc': r['metrics']['roc_auc']}
                        for r in results],
    }
    with open(os.path.join(DATA_DIR, 'best_model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    joblib.dump(best['model_obj'], os.path.join(DATA_DIR, 'best_model.pkl'))
    print(f'Guardado: best_model.pkl ({best["model_type"]}) | run_id: {best["run_id"]}')


if __name__ == '__main__':
    main()
