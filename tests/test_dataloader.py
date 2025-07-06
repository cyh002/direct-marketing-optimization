import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.dataloader import DataLoader
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def get_loader():
    """Instantiate DataLoader using resolved Hydra configuration."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"]["raw_excel_path"] = os.path.join(
        os.path.dirname(CONFIG_PATH), "data", "raw", "DataScientist_CaseStudy_Dataset.xlsx"
    )
    return DataLoader(config=cfg)


def test_load_configured_sheets():
    """DataLoader should load all sheets specified in the config."""
    loader = get_loader()
    datasets = loader.load_configured_sheets()
    assert set(datasets.keys()) == {
        "Soc_Dem",
        "Products_ActBalance",
        "Inflow_Outflow",
        "Sales_Revenues",
    }
    for df in datasets.values():
        assert isinstance(df, pd.DataFrame)


def test_create_sales_data_split():
    """Verify that clients are correctly split based on sales data."""
    loader = get_loader()
    datasets = loader.load_configured_sheets()
    with_sales, without_sales = loader.create_sales_data_split(datasets)
    assert len(with_sales["Sales_Revenues_with_sales"]) == 969
    assert len(without_sales["Sales_Revenues_without_sales"]) == 0

