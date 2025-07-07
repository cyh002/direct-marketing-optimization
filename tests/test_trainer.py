import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer, RevenueTrainer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def get_training_data():
    """Return preprocessing pipeline and training targets for CC product."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"]["raw_excel_path"] = os.path.join(
        os.path.dirname(CONFIG_PATH), "data", "raw", "DataScientist_CaseStudy_Dataset.xlsx"
    )
    cfg["preprocessing"]["enable_save"] = False
    loader = DataLoader(config=cfg)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    train_sets, _ = preprocessor.create_model_train_test_split(with_sales, test_size=0.2, random_state=42)
    merged = preprocessor.merge_datasets(train_sets, base_dataset_key="Sales_Revenues_train")
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    y_prop = merged["Sale_CC"]
    y_rev = merged["Revenue_CC"]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    return pipeline, X, y_prop, y_rev


def test_propensity_trainer(tmp_path):
    """Ensure PropensityTrainer saves model and metadata."""
    pipeline, X, y_prop, _ = get_training_data()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    model = instantiate(cfg.propensity_model)
    trainer = PropensityTrainer(model=model, preprocessor=pipeline, cv=2, output_dir=tmp_path)
    metadata = trainer.fit(X, y_prop)
    model_file = tmp_path / f"{model.__class__.__name__}.joblib"
    meta_file = tmp_path / f"{model.__class__.__name__}_metadata.json"
    assert model_file.exists()
    assert meta_file.exists()
    assert metadata.train_score is not None
    assert metadata.test_score is not None


def test_revenue_trainer(tmp_path):
    """Ensure RevenueTrainer saves model and metadata."""
    pipeline, X, _, y_rev = get_training_data()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    model = instantiate(cfg.revenue_model)
    trainer = RevenueTrainer(model=model, preprocessor=pipeline, cv=2, output_dir=tmp_path)
    metadata = trainer.fit(X, y_rev)
    model_file = tmp_path / f"{model.__class__.__name__}.joblib"
    meta_file = tmp_path / f"{model.__class__.__name__}_metadata.json"
    assert model_file.exists()
    assert meta_file.exists()
    assert metadata.train_score is not None
    assert metadata.test_score is not None
