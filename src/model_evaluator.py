"""Model evaluation utilities for training workflows."""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, List

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .logging import get_logger
from .metrics import (
    MODEL_METRIC_REGISTRY,
    ModelMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    R2Metric,
    AdjustedR2Metric,
    RMSEMetric,
)
from .mlflow_utils import MlflowConfig
import contextlib
import mlflow


class BaseModelEvaluator:
    """Common logic for model evaluators."""

    def __init__(
        self,
        metrics: Sequence[str | ModelMetric],
        mlflow_config: Optional[MlflowConfig] = None,
        run_name: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> None:
        self.metrics: List[ModelMetric] = [self._resolve(m) for m in metrics]
        self.mlflow_config = mlflow_config
        self.run_name = run_name
        self.artifact_dir = artifact_dir
        self.logger = get_logger(self.__class__.__name__)

    def _mlflow_run(self):
        if not self.mlflow_config or not self.mlflow_config.enabled:
            return contextlib.nullcontext()
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        run_name = self.run_name or self.__class__.__name__
        return mlflow.start_run(run_name=run_name)

    def _resolve(self, metric: str | ModelMetric) -> ModelMetric:
        if isinstance(metric, ModelMetric):
            return metric
        metric_cls = MODEL_METRIC_REGISTRY[metric]
        return metric_cls()  # type: ignore[return-value]

    def _log(self, values: Dict[str, float], artifacts: Sequence[str] | None = None) -> None:
        if self.mlflow_config and self.mlflow_config.enabled:
            with self._mlflow_run():
                mlflow.log_metrics(values)
                if artifacts:
                    for art in artifacts:
                        mlflow.log_artifact(art, artifact_path="plots")


class ClassifierEvaluator(BaseModelEvaluator):
    """Evaluate classification models and log metrics."""

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            metrics=[PrecisionMetric.name, RecallMetric.name, F1Metric.name],
            **kwargs,
        )
        self.threshold = threshold

    def evaluate(self, model, X, y) -> Dict[str, float]:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
        else:
            prob = model.predict(X)
        preds = (prob >= self.threshold).astype(int)
        values = {m.name: m.compute(y, preds) for m in self.metrics}

        plot_path = None
        if self.artifact_dir:
            cm = confusion_matrix(y, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plot_path = os.path.join(self.artifact_dir, "test_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()

        self._log(values, [plot_path] if plot_path else None)
        return values


class RegressorEvaluator(BaseModelEvaluator):
    """Evaluate regression models and log metrics."""

    def __init__(self, n_features: int, **kwargs) -> None:
        super().__init__(
            metrics=[R2Metric.name, AdjustedR2Metric.name, RMSEMetric.name],
            **kwargs,
        )
        self.n_features = n_features

    def _resolve(self, metric: str | ModelMetric) -> ModelMetric:
        if metric == AdjustedR2Metric.name:
            return AdjustedR2Metric(self.n_features)
        return super()._resolve(metric)

    def evaluate(self, model, X, y) -> Dict[str, float]:
        preds = model.predict(X)
        values = {m.name: m.compute(y, preds) for m in self.metrics}
        self._log(values)
        return values

