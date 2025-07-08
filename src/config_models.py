from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict, field_validator


class FeaturesConfig(BaseModel):
    numeric: List[str]
    categorical: Optional[List[str]] = None

    @field_validator("numeric")
    @classmethod
    def numeric_not_empty(cls, v: List[str]) -> List[str]:
        """Validate that numeric features list is not empty."""
        if not v:
            raise ValueError("numeric feature list cannot be empty")
        return v


class NumericImputerConfig(BaseModel):
    strategy: str = "constant"
    fill_value: Any = 0


class CategoricalImputerConfig(BaseModel):
    strategy: str = "constant"
    fill_value: str = "missing"


class OneHotConfig(BaseModel):
    handle_unknown: str = "ignore"


class CollinearityFilterConfig(BaseModel):
    """Settings for multicollinearity removal."""

    enabled: bool = False
    corr_threshold: float = 0.8
    vif_threshold: float = 5.0


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
    enable_save: bool = False
    save_path: str = "./preprocessed"
    collinearity_filter: CollinearityFilterConfig = CollinearityFilterConfig()


class DataConfig(BaseModel):
    raw_excel_path: str
    sheets: List[str]

    @field_validator("sheets")
    @classmethod
    def sheets_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one sheet is specified."""
        if not v:
            raise ValueError("sheets list cannot be empty")
        return v


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

    @field_validator("k_folds")
    @classmethod
    def k_folds_positive(cls, v: int) -> int:
        """k_folds must be a positive integer."""
        if v <= 0:
            raise ValueError("k_folds must be greater than 0")
        return v

    @field_validator("sample_fraction")
    @classmethod
    def sample_fraction_range(cls, v: float) -> float:
        """Validate sample_fraction is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        return v


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


class ModelConfig(BaseModel):
    """Configuration for an estimator."""

    model_config = ConfigDict(extra="allow")
    _target_: str


class ConfigSchema(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    propensity_model: ModelConfig
    revenue_model: ModelConfig
    products: List[str]
    targets: Dict[str, str]
    training: TrainingConfig = TrainingConfig()
    mlflow: MlflowConfig = MlflowConfig()
    inference: InferenceConfig = InferenceConfig()

    class Config:
        extra = "ignore"
