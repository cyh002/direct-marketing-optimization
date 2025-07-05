"""Entry point for the direct marketing optimization pipeline."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.logging import get_logger


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run data loading and preprocessing steps.

    This function loads the datasets defined in the project configuration,
    performs basic preprocessing, and prepares the feature pipeline. Further
    steps like model training and optimization will be implemented later.

    Args:
        cfg: Hydra configuration object.
    """
    logger = get_logger(__name__)
    logger.info("Starting direct marketing optimization pipeline")

    config_path = Path(get_original_cwd()) / "conf" / "config.yaml"

    # Load datasets
    loader = DataLoader(config_path=str(config_path))
    datasets = loader.load_configured_sheets()
    logger.info("Loaded datasets: %s", list(datasets))

    # Split into clients with/without sales data
    with_sales, without_sales = loader.create_sales_data_split(datasets)
    logger.info(
        "Created sales split. With sales: %d, Without sales: %d",
        len(with_sales),
        len(without_sales),
    )

    # Set up preprocessor
    preprocessor = Preprocessor(loader.get_config())
    train_sets, test_sets = preprocessor.create_model_train_test_split(with_sales)
    logger.info(
        "Train datasets: %s, Test datasets: %s",
        list(train_sets),
        list(test_sets),
    )

    base_train_key = next(key for key in train_sets if "Sales_Revenues" in key)
    base_test_key = next(key for key in test_sets if "Sales_Revenues" in key)
    merged_train = preprocessor.merge_datasets(train_sets, base_dataset_key=base_train_key)
    merged_test = preprocessor.merge_datasets(test_sets, base_dataset_key=base_test_key)
    logger.info("Merged train shape: %s, test shape: %s", merged_train.shape, merged_test.shape)

    numeric_features, categorical_features = loader.get_feature_lists()
    pipeline = preprocessor.create_preprocessing_pipeline(numeric_features, categorical_features)
    preprocessor.analyze_data_quality(merged_train, show_columns=False)

    logger.info("Preprocessing pipeline ready: %s", pipeline)
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
