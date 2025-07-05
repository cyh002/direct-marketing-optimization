from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

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

class PreprocessingConfig(BaseModel):
    features: FeaturesConfig
    categorical: Optional[List[str]] = None
    numeric_imputer: NumericImputerConfig = NumericImputerConfig()
    categorical_imputer: CategoricalImputerConfig = CategoricalImputerConfig()
    onehot: OneHotConfig = OneHotConfig()

class DataConfig(BaseModel):
    raw_excel_path: str
    sheets: List[str]

class ConfigSchema(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    products: List[str]
    targets: Dict[str, str]

