"""Training utilities for propensity and revenue models."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import contextlib

import mlflow

import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .logging import get_logger
from .config_models import MlflowConfig


class ModelMetadata(BaseModel):
    """Metadata about a trained model."""

    model_name: str
    params: Dict[str, Any]
    train_score: float
    test_score: float


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Pipeline,
        scoring: str,
        cv: int = 5,
        output_dir: str = "outputs/models",
        mlflow_config: Optional[MlflowConfig] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.scoring = scoring
        self.cv = cv
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)
        self.pipeline: Optional[Pipeline] = None
        self.train_score: Optional[float] = None
        self.test_score: Optional[float] = None
        self.mlflow_config = mlflow_config
        self.run_name = run_name

    def _mlflow_run(self):
        if not self.mlflow_config or not self.mlflow_config.enabled:
            return contextlib.nullcontext()
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        run_name = self.run_name or self.model.__class__.__name__
        return mlflow.start_run(run_name=run_name)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelMetadata:
        """Fit the model and return metadata."""

    def save(self, metadata: ModelMetadata) -> None:
        """Persist the trained model and metadata."""
        os.makedirs(self.output_dir, exist_ok=True)
        model_name = metadata.model_name
        model_path = os.path.join(self.output_dir, f"{model_name}.joblib")
        joblib.dump(self.pipeline, model_path)
        meta_path = os.path.join(self.output_dir, f"{model_name}_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        self.logger.info("Model saved to %s", model_path)
        self.logger.info("Metadata saved to %s", meta_path)


class Trainer(BaseTrainer):
    """Generic trainer for scikit-learn models."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelMetadata:
        self.pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("model", self.model)]
        )
        with self._mlflow_run():
            self.logger.info(
                "Starting cross validation with scoring=%s and %d folds",
                self.scoring,
                self.cv,
            )
            cv_results = cross_validate(
                self.pipeline,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_train_score=True,
            )
            # Log per-fold metrics for detailed loss curves
            for idx, (train_fold, test_fold) in enumerate(
                zip(cv_results["train_score"], cv_results["test_score"])
            ):
                self.logger.debug(
                    "Fold %d: train_score=%.4f, test_score=%.4f",
                    idx,
                    train_fold,
                    test_fold,
                )
                if self.mlflow_config and self.mlflow_config.enabled:
                    mlflow.log_metric("train_score_fold", train_fold, step=idx)
                    mlflow.log_metric("test_score_fold", test_fold, step=idx)
            self.train_score = float(cv_results["train_score"].mean())
            self.test_score = float(cv_results["test_score"].mean())
            self.logger.info(
                "Cross validation completed: train_score=%.4f, test_score=%.4f",
                self.train_score,
                self.test_score,
            )
            self.pipeline.fit(X, y)
            metadata = ModelMetadata(
                model_name=self.model.__class__.__name__,
                params=self.model.get_params(),
                train_score=self.train_score,
                test_score=self.test_score,
            )
            if self.mlflow_config and self.mlflow_config.enabled:
                mlflow.log_params(metadata.params)
                mlflow.log_metrics(
                    {
                        "train_score": self.train_score,
                        "test_score": self.test_score,
                    }
                )
                mlflow.sklearn.log_model(self.pipeline, "model")
            self.save(metadata)
            return metadata


class PropensityTrainer(Trainer):
    """Trainer specialized for propensity models."""

    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Pipeline,
        scoring: str = "f1",
        cv: int = 5,
        output_dir: str = "outputs/models",
        mlflow_config: Optional[MlflowConfig] = None,
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            model,
            preprocessor,
            scoring=scoring,
            cv=cv,
            output_dir=output_dir,
            mlflow_config=mlflow_config,
            run_name=run_name,
        )


class RevenueTrainer(Trainer):
    """Trainer specialized for revenue models."""

    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Pipeline,
        scoring: str = "neg_root_mean_squared_error",
        cv: int = 5,
        output_dir: str = "outputs/models",
        mlflow_config: Optional[MlflowConfig] = None,
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            model,
            preprocessor,
            scoring=scoring,
            cv=cv,
            output_dir=output_dir,
            mlflow_config=mlflow_config,
            run_name=run_name,
        )
