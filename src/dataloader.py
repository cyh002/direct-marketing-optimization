"""Data loading utilities."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config_loader import ConfigLoader
from .logging import get_logger


class DataLoader:
    """Load Excel datasets and provide business logic splits."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.config_loader = ConfigLoader(config_path=config_path, config=config)
        self.config = self.config_loader.get_config()

    def load_excel_datasets(self, excel_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load all sheets from an Excel workbook.

        Args:
            excel_path: Optional path to the Excel file. When ``None`` the path
                from the configuration is used.

        Returns:
            Mapping of sheet name to loaded :class:`pandas.DataFrame`.

        Raises:
            FileNotFoundError: If the provided file does not exist.
            ValueError: If the file cannot be read.
        """
        if excel_path is None:
            excel_path = self.config_loader.resolve_path(self.config.data.raw_excel_path)
        else:
            excel_path = self.config_loader.resolve_path(excel_path)

        self.logger.info("Loading Excel file from %s", excel_path)

        try:
            xls = pd.ExcelFile(excel_path, engine="openpyxl")
            datasets = {
                sheet: pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
                for sheet in xls.sheet_names
            }
        except FileNotFoundError as exc:  # pragma: no cover - I/O errors
            raise FileNotFoundError(
                f"Excel file not found at: {excel_path}"
            ) from exc
        except Exception as exc:  # pragma: no cover - general catch
            raise ValueError(f"Error loading Excel file: {exc}") from exc

        datasets.pop("Description", None)
        return datasets

    def load_configured_sheets(
        self, excel_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load only the sheets defined in the configuration file.

        Args:
            excel_path: Optional path to override the configured Excel file.

        Returns:
            Mapping with sheet names as keys and :class:`pandas.DataFrame` as
            values.
        """
        all_datasets = self.load_excel_datasets(excel_path)
        configured_datasets: Dict[str, pd.DataFrame] = {}
        for sheet in self.config.data.sheets:
            if sheet in all_datasets:
                configured_datasets[sheet] = all_datasets[sheet]
                self.logger.debug("Loaded sheet %s", sheet)
            else:
                self.logger.warning("Configured sheet '%s' not found", sheet)
        return configured_datasets

    def create_sales_data_split(
        self, datasets: Dict[str, pd.DataFrame], sales_key: str = "Sales_Revenues"
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split datasets based on whether clients have sales data.

        Args:
            datasets: Mapping of dataset names to dataframes.
            sales_key: Name of the sales dataset containing the ``Client`` column.

        Returns:
            Two dictionaries with ``_with_sales`` and ``_without_sales`` suffixes
            indicating which clients appear in the sales data.
        """
        if sales_key not in datasets:
            raise ValueError(f"Sales dataset '{sales_key}' not found in datasets")

        sales_clients = set(datasets[sales_key]["Client"].unique())
        datasets_with_sales: Dict[str, pd.DataFrame] = {}
        datasets_without_sales: Dict[str, pd.DataFrame] = {}

        for name, df in datasets.items():
            if "Client" in df.columns:
                with_sales_mask = df["Client"].isin(sales_clients)
                with_sales_df = df[with_sales_mask].copy()
                without_sales_df = df[~with_sales_mask].copy()
                datasets_with_sales[f"{name}_with_sales"] = with_sales_df
                datasets_without_sales[f"{name}_without_sales"] = without_sales_df
                self._log_split_summary(name, df, with_sales_df, without_sales_df)
        return datasets_with_sales, datasets_without_sales

    def _log_split_summary(
        self, name: str, original_df: pd.DataFrame, with_sales_df: pd.DataFrame, without_sales_df: pd.DataFrame
    ) -> None:
        """Log the row counts after splitting a dataset."""
        total_rows = len(original_df)
        self.logger.debug(
            "%s - total: %d, with_sales: %d, without_sales: %d",
            name,
            total_rows,
            len(with_sales_df),
            len(without_sales_df),
        )

    def get_feature_lists(self) -> Tuple[List[str], List[str]]:
        numeric_features = self.config.preprocessing.features.numeric
        categorical_features = (
            self.config.preprocessing.features.categorical or self.config.preprocessing.categorical
        )
        return numeric_features, categorical_features

    def get_products(self) -> List[str]:
        return self.config.products

    def get_targets(self) -> Dict[str, str]:
        return self.config.targets

    def get_config(self) -> Dict:
        return self.config.model_dump()


def load_datasets_from_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper to load datasets from a config path."""
    loader = DataLoader(config_path=config_path)
    return loader.load_configured_sheets()


def load_and_create_sales_split(
    config_path: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], DataLoader]:
    """Load datasets and create sales splits in one call."""
    loader = DataLoader(config_path=config_path)
    datasets = loader.load_configured_sheets()
    with_sales, without_sales = loader.create_sales_data_split(datasets)
    return with_sales, without_sales, loader

