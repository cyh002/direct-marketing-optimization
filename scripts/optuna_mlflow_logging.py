"""Example Optuna study that logs loss curves to MLflow."""
from __future__ import annotations

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np


def objective(trial: optuna.Trial) -> float:
    """Train a simple classifier and log loss per epoch."""
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    epochs = 5
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=lr,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    for epoch in range(epochs):
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_valid)
        loss = log_loss(y_valid, preds)
        mlflow.log_metric("loss", loss, step=epoch)
        trial.report(loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return loss


if __name__ == "__main__":
    data = load_iris()
    X_train, X_valid, y_train, y_valid = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    mlflow_cb = MLflowCallback(tracking_uri="mlruns", metric_name="loss")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3, callbacks=[mlflow_cb])
    print("Best value:", study.best_value)
