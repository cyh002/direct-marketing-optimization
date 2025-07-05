import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer, RevenueTrainer


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    loader = DataLoader(config=config_dict)
    loader.config_loader.base_dir = os.path.join(get_original_cwd(), "conf")

    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    train_sets, _ = preprocessor.create_model_train_test_split(
        with_sales, test_size=0.2, random_state=cfg.training.random_seed
    )
    merged = preprocessor.merge_datasets(train_sets, base_dataset_key="Sales_Revenues_train")
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    y_propensity = merged[f"{cfg.targets.propensity}_CC"]
    y_revenue = merged[f"{cfg.targets.revenue}_CC"]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)

    mlflow_cfg = loader.config.mlflow

    prop_trainer = PropensityTrainer(
        model=LogisticRegression(max_iter=200),
        preprocessor=pipeline,
        scoring=cfg.training.propensity_scoring,
        cv=cfg.training.k_folds,
        output_dir=cfg.training.output_dir,
        mlflow_config=mlflow_cfg,
    )
    prop_trainer.fit(X, y_propensity)

    rev_trainer = RevenueTrainer(
        model=RandomForestRegressor(),
        preprocessor=pipeline,
        scoring=cfg.training.revenue_scoring,
        cv=cfg.training.k_folds,
        output_dir=cfg.training.output_dir,
        mlflow_config=mlflow_cfg,
    )
    rev_trainer.fit(X, y_revenue)


if __name__ == "__main__":
    main()
