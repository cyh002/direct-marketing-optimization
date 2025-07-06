import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def get_loader_and_preprocessor():
    """Create DataLoader and Preprocessor for tests using resolved config."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"]["raw_excel_path"] = os.path.join(
        os.path.dirname(CONFIG_PATH), "data", "raw", "DataScientist_CaseStudy_Dataset.xlsx"
    )
    loader = DataLoader(config=cfg)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    return loader, preprocessor, with_sales


def test_model_train_test_split():
    """Verify deterministic splitting of clients into train and test sets."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, test = preprocessor.create_model_train_test_split(with_sales, test_size=0.2, random_state=42)
    assert len(train["Sales_Revenues_train"]) == 775
    assert len(test["Sales_Revenues_test"]) == 194


def test_merge_datasets():
    """Ensure datasets merge correctly on the Client key."""
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
    """Check that preprocessing pipeline can be constructed."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    numeric, categorical = loader.get_feature_lists()
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    assert hasattr(pipeline, "transform")


def test_preprocessing_no_missing(tmp_path):
    """Processed features should contain no NaN values."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, _ = preprocessor.create_model_train_test_split(with_sales)
    merged = preprocessor.merge_datasets(train, base_dataset_key="Sales_Revenues_train")
    numeric, categorical = loader.get_feature_lists()
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    transformed = pipeline.fit_transform(merged[numeric + categorical])
    assert not pd.isnull(transformed).any()

