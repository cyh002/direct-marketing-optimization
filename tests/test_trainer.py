import os
from hydra import compose, initialize_config_dir
from sklearn.linear_model import LogisticRegression
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "conf")


def get_training_data():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        compose(config_name="config")
    config_file = os.path.join(CONFIG_PATH, "config.yaml")
    loader = DataLoader(config_path=config_file)
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    train_sets, _ = preprocessor.create_model_train_test_split(with_sales, test_size=0.2, random_state=42)
    merged = preprocessor.merge_datasets(train_sets, base_dataset_key="Sales_Revenues_train")
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    y = merged["Sale_CC"]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    return pipeline, X, y


def test_propensity_trainer(tmp_path):
    pipeline, X, y = get_training_data()
    model = LogisticRegression(max_iter=200)
    trainer = PropensityTrainer(model=model, preprocessor=pipeline, cv=2, output_dir=tmp_path)
    metadata = trainer.fit(X, y)
    model_file = tmp_path / f"{model.__class__.__name__}.joblib"
    meta_file = tmp_path / f"{model.__class__.__name__}_metadata.json"
    assert model_file.exists()
    assert meta_file.exists()
    assert metadata.train_score is not None
    assert metadata.test_score is not None
