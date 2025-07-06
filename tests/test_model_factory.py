import os
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from src.model_factory import ModelFactory

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "conf")


def test_load_logistic_regression():
    model = ModelFactory.from_config("propensity", "LogisticRegression", CONFIG_DIR)
    assert isinstance(model, LogisticRegression)


def test_load_random_forest():
    model = ModelFactory.from_config("revenue", "RandomForest", CONFIG_DIR)
    assert isinstance(model, RandomForestRegressor)


def test_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelFactory.from_config("propensity", "NonExistent", str(tmp_path))
