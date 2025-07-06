# Direct Marketing Optimization

This project predicts customer propensity and revenue to optimize marketing campaigns. It uses Hydra for configuration, Optuna for hyperparameter search, and MLflow for experiment tracking.

## Logging Loss Curves with Optuna

The script [`scripts/optuna_mlflow_logging.py`](scripts/optuna_mlflow_logging.py) shows how to log loss values to MLflow during an Optuna optimization. Each training epoch is logged as a metric so you can visualize loss curves in the MLflow UI.
