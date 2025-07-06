import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer, RevenueTrainer
from src.inference import PropensityInference, RevenueInference


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def prepare_data():
    """Return fitted preprocessing pipeline and sample data for CC product."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"]["raw_excel_path"] = os.path.join(
        os.path.dirname(CONFIG_PATH), "data", "raw", "DataScientist_CaseStudy_Dataset.xlsx"
    )
    loader = DataLoader(config=cfg)
    datasets = loader.load_configured_sheets()
    preprocessor = Preprocessor(loader.get_config())
    merged = preprocessor.merge_datasets(datasets, base_dataset_key="Sales_Revenues")
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    y_prop = merged["Sale_CC"]
    y_rev = merged["Revenue_CC"]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    return pipeline, X, y_prop, y_rev


def test_inference_workflow(tmp_path):
    """Train models, run inference and validate prediction ranges."""
    pipeline, X, y_prop, y_rev = prepare_data()
    prop_dir = tmp_path / "propensity" / "CC"
    rev_dir = tmp_path / "revenue" / "CC"
    os.makedirs(prop_dir, exist_ok=True)
    os.makedirs(rev_dir, exist_ok=True)

    PropensityTrainer(
        model=LogisticRegression(max_iter=100),
        preprocessor=pipeline,
        cv=2,
        output_dir=prop_dir,
    ).fit(X, y_prop)

    RevenueTrainer(
        model=RandomForestRegressor(n_estimators=10),
        preprocessor=pipeline,
        cv=2,
        output_dir=rev_dir,
    ).fit(X, y_rev)

    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["training"]["train_enabled"] = False
    cfg["training"]["load_model_path"] = str(tmp_path)
    cfg["inference"] = {"output_dir": str(tmp_path)}
    cfg["products"] = ["CC"]

    prop_inf = PropensityInference(config=cfg)
    rev_inf = RevenueInference(config=cfg)

    prop_preds = prop_inf.predict(X)
    rev_preds = rev_inf.predict(X)

    prop_path = prop_inf.save(prop_preds, "prop.csv")
    rev_path = rev_inf.save(rev_preds, "rev.csv")

    assert os.path.exists(prop_path)
    assert os.path.exists(rev_path)
    prop_df = pd.read_csv(prop_path)
    rev_df = pd.read_csv(rev_path)
    assert "Client" in prop_df.columns
    assert "Client" in rev_df.columns
    assert prop_df.shape[0] == X.shape[0]
    assert rev_df.shape[0] == X.shape[0]
    assert prop_preds.shape[0] == X.shape[0]
    assert rev_preds.shape[0] == X.shape[0]
    assert (prop_preds > 0).all().all()
    assert (prop_preds <= 1.0).all().all()
    assert (rev_preds >= 0).all().all()


