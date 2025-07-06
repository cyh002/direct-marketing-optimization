from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict

class FeaturesConfig(BaseModel):
    numeric: List[str]
    categorical: Optional[List[str]] = None

class NumericImputerConfig(BaseModel):
    strategy: str = "constant"
    fill_value: Any = 0

class CategoricalImputerConfig(BaseModel):
    strategy: str = "constant"
    fill_value: str = "missing"

class OneHotConfig(BaseModel):
    handle_unknown: str = "ignore"


class TrainTestSplitConfig(BaseModel):
    """Configuration for train-test split."""

    test_size: float = 0.2
    random_state: int = 42

class PreprocessingConfig(BaseModel):
    features: FeaturesConfig
    categorical: Optional[List[str]] = None
    numeric_imputer: NumericImputerConfig = NumericImputerConfig()
    categorical_imputer: CategoricalImputerConfig = CategoricalImputerConfig()
    onehot: OneHotConfig = OneHotConfig()
    train_test_split: TrainTestSplitConfig = TrainTestSplitConfig()

class DataConfig(BaseModel):
    raw_excel_path: str
    sheets: List[str]


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    model_config = ConfigDict(protected_namespaces=("model_validate",))

    k_folds: int = 5
    random_seed: int = 42
    propensity_scoring: str = "f1"
    revenue_scoring: str = "neg_root_mean_squared_error"
    model_dump_path: str = "./outputs/models"
    load_model_path: Optional[str] = None
    sample_fraction: float = 1.0
    train_enabled: bool = True


class MlflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "direct_marketing"


class InferenceConfig(BaseModel):
    """Configuration for inference output."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "./data/inference"
    propensity_file: str = "propensity_predictions.csv"
    revenue_file: str = "revenue_predictions.csv"

class ConfigSchema(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    products: List[str]
    targets: Dict[str, str]
    training: TrainingConfig = TrainingConfig()
    mlflow: MlflowConfig = MlflowConfig()
    inference: InferenceConfig = InferenceConfig()

    class Config:
        extra = "ignore"

