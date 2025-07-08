import pandas as pd
import numpy as np
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
        os.path.dirname(CONFIG_PATH),
        "data",
        "raw",
        "DataScientist_CaseStudy_Dataset.xlsx",
    )
    cfg["preprocessing"]["enable_save"] = False
    loader = DataLoader(config=cfg)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    return loader, preprocessor, with_sales


def test_model_train_test_split():
    """Verify deterministic splitting of clients into train and test sets."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, test = preprocessor.create_model_train_test_split(
        with_sales, test_size=0.2, random_state=42
    )
    assert len(train["Sales_Revenues_train"]) == 775
    assert len(test["Sales_Revenues_test"]) == 194


def test_model_train_test_split_from_config():
    """Split parameters should default to config values when not provided."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"]["raw_excel_path"] = os.path.join(
        os.path.dirname(CONFIG_PATH),
        "data",
        "raw",
        "DataScientist_CaseStudy_Dataset.xlsx",
    )
    cfg["preprocessing"]["train_test_split"]["test_size"] = 0.25
    loader = DataLoader(config=cfg)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    train, test = preprocessor.create_model_train_test_split(with_sales)
    assert len(test["Sales_Revenues_test"]) == 243


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


def test_train_test_split_saved(tmp_path):
    """Train-test split should be saved when enabled in config."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    preprocessor.config.preprocessing.enable_save = True
    preprocessor.config.preprocessing.save_path = str(tmp_path)
    preprocessor.create_model_train_test_split(
        with_sales, test_size=0.2, random_state=42
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    assert train_path.exists()
    assert test_path.exists()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert set(train_df.columns) == set(test_df.columns)


def test_remove_multicollinearity():
    """Numeric features should be reduced when thresholds are low."""
    loader, preprocessor, with_sales = get_loader_and_preprocessor()
    train, _ = preprocessor.create_model_train_test_split(with_sales)
    merged = preprocessor.merge_datasets(train, base_dataset_key="Sales_Revenues_train")
    numeric, _ = loader.get_feature_lists()
    filtered = preprocessor.remove_multicollinearity(
        merged, numeric, corr_threshold=0.1, vif_threshold=1.0
    )
    assert set(filtered).issubset(set(numeric))
    assert len(filtered) < len(numeric)


def test_remove_multicollinearity_drops_single_from_pair():
    """Only one of a highly correlated pair should be removed."""
    _loader, preprocessor, _ = get_loader_and_preprocessor()
    np.random.seed(0)
    n = 200
    x1 = np.random.randn(n)
    x2 = x1 + np.random.normal(scale=0.01, size=n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    features = ["x1", "x2", "x3", "x4"]
    remaining = preprocessor.remove_multicollinearity(
        df,
        features,
        corr_threshold=0.8,
        vif_threshold=5.0,
    )
    assert "x1" in remaining or "x2" in remaining
    assert not ("x1" not in remaining and "x2" not in remaining)
