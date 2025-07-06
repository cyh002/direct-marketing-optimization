import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from src.model_factory import ModelFactory

from src.dataloader import DataLoader
from src.evaluator import Evaluator
from src.inference import PropensityInference, RevenueInference
from src.model_loader import ModelLoader
from src.optimizer import Optimizer
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer, RevenueTrainer
from src.logging import get_logger, setup_logging
from src.mlflow_utils import ensure_mlflow_server

setup_logging()
logger = get_logger(__name__)

import numpy as np
import pandas as pd

def run_inference(
    cfg: DictConfig,
    preprocessor: Preprocessor,
    loader: DataLoader,
    datasets: dict,
    config_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Run propensity and revenue inference."""
    logger.info("Merging datasets for inference")
    inference_df = preprocessor.merge_datasets(
        datasets, base_dataset_key="Sales_Revenues"
    )
    numeric, categorical = loader.get_feature_lists()
    X_inf = inference_df[numeric + categorical]
    X_inf.index = inference_df["Client"]

    logger.info("Running propensity and revenue inference")
    prop_inf = PropensityInference(config=config_dict)
    rev_inf = RevenueInference(config=config_dict)
    prop_preds = prop_inf.predict(X_inf)
    rev_preds = rev_inf.predict(X_inf)

    logger.info("Saving inference results")
    prop_inf.save(prop_preds, cfg.inference.propensity_file)
    rev_inf.save(rev_preds, cfg.inference.revenue_file)
    return prop_preds, rev_preds, X_inf.index


def run_optimization(
    cfg: DictConfig, prop_preds: pd.DataFrame, rev_preds: pd.DataFrame
) -> np.ndarray:
    """Optimize the marketing offers."""
    logger.info("Running optimization")
    expected = prop_preds.values * rev_preds.values
    optimizer = Optimizer(contact_limit=cfg.optimization.contact_limit)
    return optimizer.solve(expected)


def run_evaluation(
    cfg: DictConfig,
    selection: np.ndarray,
    prop_preds: pd.DataFrame,
    rev_preds: pd.DataFrame,
    config_dict: dict,
) -> dict:
    """Evaluate optimized offers using configured metrics."""
    logger.info("Evaluating optimization results")
    evaluator = Evaluator(
        config=config_dict, cost_per_contact=cfg.evaluation.cost_per_contact
    )
    results = evaluator.evaluate(selection, prop_preds.values, rev_preds.values)
    for name, value in results.items():
        logger.info("%s: %.4f", name, value)
    return results


def save_offers(
    cfg: DictConfig,
    selection: np.ndarray,
    prop_preds: pd.DataFrame,
    rev_preds: pd.DataFrame,
    client_index: pd.Index,
) -> pd.DataFrame:
    """Persist the optimized offer list to disk."""
    logger.info("Saving optimized offers")
    selection_df = pd.DataFrame(selection, index=client_index, columns=cfg.products)
    result_rows = []
    for product in cfg.products:
        mask = selection_df[product] == 1
        if mask.any():
            df_sel = pd.DataFrame(
                {
                    "Client": selection_df.index[mask],
                    "product": product,
                    "expected_revenue": rev_preds.loc[
                        mask, f"expected_revenue_{product}"
                    ].values,
                    "probability": prop_preds.loc[
                        mask, f"probability_{product}"
                    ].values,
                }
            )
            result_rows.append(df_sel)

    results_df = pd.concat(result_rows, ignore_index=True)
    save_path = cfg.optimization.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    return results_df


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting main workflow")
    ensure_mlflow_server(cfg.mlflow, logger=logger)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    loader = DataLoader(config=config_dict)
    config_base_dir = os.path.join(get_original_cwd(), "conf")
    loader.config_loader.base_dir = config_base_dir

    logger.info("Loading datasets")
    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    logger.info("Created sales data split")
    preprocessor = Preprocessor(loader.get_config())
    train_sets, _ = preprocessor.create_model_train_test_split(
        with_sales, test_size=cfg.preprocessing.train_test_split.test_size, random_state=cfg.preprocessing.train_test_split.random_state
    )
    logger.info("Created train/test split")
    merged = preprocessor.merge_datasets(
        train_sets, base_dataset_key="Sales_Revenues_train"
    )
    logger.info("Merged training datasets")
    if cfg.training.sample_fraction < 1.0:
        merged = merged.sample(
            frac=cfg.training.sample_fraction, random_state=cfg.training.random_seed
        )
        logger.info(
            "Subsampled training data to %.2f%%",
            cfg.training.sample_fraction * 100,
        )
    logger.info("Training dataset shape: %s", merged.shape)
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)
    logger.info("Preprocessing pipeline created")

    mlflow_cfg = loader.config.mlflow

    def train_for_product(product: str) -> None:
        logger.info("Processing product %s", product)
        y_propensity = merged[f"{cfg.targets.propensity}_{product}"]
        y_revenue = merged[f"{cfg.targets.revenue}_{product}"]
        prop_out = os.path.join(cfg.training.model_dump_path, "propensity", product)
        rev_out = os.path.join(cfg.training.model_dump_path, "revenue", product)
        os.makedirs(prop_out, exist_ok=True)
        os.makedirs(rev_out, exist_ok=True)

        if cfg.training.train_enabled:
            logger.info("Training models for %s", product)
            prop_model = ModelFactory.from_config(
                "propensity",
                cfg.training.propensity_model,
                config_base_dir,
            )
            prop_trainer = PropensityTrainer(
                model=prop_model,
                preprocessor=pipeline,
                scoring=cfg.training.propensity_scoring,
                cv=cfg.training.k_folds,
                output_dir=prop_out,
                mlflow_config=mlflow_cfg,
                run_name=f"propensity-{product}",
            )
            prop_trainer.fit(X, y_propensity)

            rev_model = ModelFactory.from_config(
                "revenue",
                cfg.training.revenue_model,
                config_base_dir,
            )
            rev_trainer = RevenueTrainer(
                model=rev_model,
                preprocessor=pipeline,
                scoring=cfg.training.revenue_scoring,
                cv=cfg.training.k_folds,
                output_dir=rev_out,
                mlflow_config=mlflow_cfg,
                run_name=f"revenue-{product}",
            )
            rev_trainer.fit(X, y_revenue)
            logger.info("Finished training models for %s", product)
        else:
            logger.info("Loading pretrained models for %s", product)
            loader = ModelLoader(config=config_dict)
            loader.load_model("propensity", product)
            loader.load_model("revenue", product)
    # ---------- Load and preprocess data ----------
    with ThreadPoolExecutor(max_workers=len(cfg.products)) as executor:
        executor.map(train_for_product, cfg.products)

    logger.info("Starting inference")
    prop_preds, rev_preds, client_index = run_inference(
        cfg, preprocessor, loader, datasets, config_dict
    )

    logger.info("Starting optimization")
    selection = run_optimization(cfg, prop_preds, rev_preds)

    logger.info("Starting evaluation")
    run_evaluation(cfg, selection, prop_preds, rev_preds, config_dict)

    logger.info("Saving optimized offers")
    save_offers(cfg, selection, prop_preds, rev_preds, client_index)


if __name__ == "__main__":
    main()
