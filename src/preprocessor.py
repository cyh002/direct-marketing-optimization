"""Data preprocessing utilities."""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from .config_models import ConfigSchema
from .logging import get_logger
import pandas as pd


class Preprocessor:
    """Prepare data for modeling and feature engineering."""

    def __init__(self, config: Dict | ConfigSchema) -> None:
        if isinstance(config, ConfigSchema):
            self.config = config
        else:
            self.config = ConfigSchema(**config)
        self.logger = get_logger(self.__class__.__name__)

    def create_model_train_test_split(
        self,
        datasets_with_sales: Dict[str, pd.DataFrame],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        sales_dataset_key = next((k for k in datasets_with_sales if "Sales_Revenues" in k), None)
        if not sales_dataset_key:
            raise ValueError("No Sales_Revenues dataset found in datasets_with_sales")

        all_clients = datasets_with_sales[sales_dataset_key]["Client"].unique()
        train_clients, test_clients = train_test_split(all_clients, test_size=test_size, random_state=random_state)

        self.logger.info("Model split: %d train clients, %d test clients", len(train_clients), len(test_clients))

        datasets_train: Dict[str, pd.DataFrame] = {}
        datasets_test: Dict[str, pd.DataFrame] = {}
        for name, df in datasets_with_sales.items():
            if "Client" in df.columns:
                train_df = df[df["Client"].isin(train_clients)].copy()
                test_df = df[df["Client"].isin(test_clients)].copy()
                base_name = name.replace("_with_sales", "")
                datasets_train[f"{base_name}_train"] = train_df
                datasets_test[f"{base_name}_test"] = test_df
        return datasets_train, datasets_test

    def merge_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        base_dataset_key: str,
        join_key: str = "Client",
        datasets_to_merge: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if base_dataset_key not in datasets:
            raise ValueError(f"Base dataset '{base_dataset_key}' not found in datasets")

        merged_df = datasets[base_dataset_key].copy()
        if datasets_to_merge is None:
            datasets_to_merge = [k for k in datasets.keys() if k != base_dataset_key]

        if join_key not in merged_df.columns:
            raise ValueError(f"Join key '{join_key}' not found in base dataset")

        for dataset_name in datasets_to_merge:
            if dataset_name in datasets:
                merged_df = self._merge_single_dataset(merged_df, datasets[dataset_name], dataset_name, join_key)
            else:
                self.logger.warning("Dataset %s not found for merge", dataset_name)

        return merged_df

    def _merge_single_dataset(
        self,
        merged_df: pd.DataFrame,
        df_to_merge: pd.DataFrame,
        dataset_name: str,
        join_key: str,
    ) -> pd.DataFrame:
        if join_key in df_to_merge.columns:
            before_shape = merged_df.shape
            merged_df = merged_df.merge(df_to_merge, on=join_key, how="left", suffixes=("", "_dup"))
            after_shape = merged_df.shape
            self.logger.debug("Merged %s: %s -> %s", dataset_name, before_shape, after_shape)
            duplicate_cols = [col for col in merged_df.columns if col.endswith("_dup")]
            if duplicate_cols:
                merged_df = merged_df.drop(columns=duplicate_cols)
        else:
            self.logger.warning("Join key '%s' not found in %s", join_key, dataset_name)
        return merged_df

    def create_preprocessing_pipeline(
        self, numeric_features: List[str], categorical_features: List[str]
    ) -> ColumnTransformer:
        preprocess_config = self.config.preprocessing
        numeric_transformer = SimpleImputer(
            strategy=preprocess_config.numeric_imputer.strategy,
            fill_value=preprocess_config.numeric_imputer.fill_value,
        )
        categorical_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy=preprocess_config.categorical_imputer.strategy,
                        fill_value=preprocess_config.categorical_imputer.fill_value,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown=preprocess_config.onehot.handle_unknown)),
            ]
        )
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])
        return preprocessor

    def analyze_data_quality(
        self, df: pd.DataFrame, show_columns: bool = True, max_missing_display: int = 20
    ) -> None:
        self.logger.info("Dataset shape: %s", df.shape)
        if show_columns:
            for i, col in enumerate(df.columns, 1):
                self.logger.debug("%02d. %s", i, col)
        self._analyze_missing_values(df, max_missing_display)
        self._analyze_data_types(df)
        self._analyze_data_issues(df)

    def _analyze_missing_values(self, df: pd.DataFrame, max_missing_display: int) -> None:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            self.logger.info("Found %d columns with missing values", len(missing_data))
            display_missing = missing_data.head(max_missing_display)
            for col, count in display_missing.items():
                percentage = (count / len(df)) * 100
                self.logger.debug("%s: %d (%.1f%%)", col, count, percentage)
        else:
            self.logger.info("No missing values found")

    def _analyze_data_types(self, df: pd.DataFrame) -> None:
        dtype_summary = df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            self.logger.debug("%s: %d columns", dtype, count)

    def _analyze_data_issues(self, df: pd.DataFrame) -> None:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning("%d duplicate rows found", duplicates)
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            self.logger.warning("Constant columns: %s", constant_cols)

