import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def get_loader_and_preprocessor():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    config_file = os.path.join(CONFIG_PATH, "config.yaml")
    loader = DataLoader(config_path=config_file)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    return loader, preprocessor, with_sales


def test_model_train_test_split():
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, test = preprocessor.create_model_train_test_split(with_sales, test_size=0.2, random_state=42)
    assert len(train["Sales_Revenues_train"]) == 775
    assert len(test["Sales_Revenues_test"]) == 194


def test_merge_datasets():
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, _ = preprocessor.create_model_train_test_split(with_sales)
    merged = preprocessor.merge_datasets(
        train,
        base_dataset_key="Sales_Revenues_train",
        join_key="Client",
    )
    assert merged.shape[0] == len(train["Sales_Revenues_train"])
    assert "Client" in merged.columns


def test_create_preprocessing_pipeline():
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    numeric, categorical = loader.get_feature_lists()
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    assert hasattr(pipeline, "transform")

