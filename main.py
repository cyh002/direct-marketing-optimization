import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.dataloader import DataLoader
from src.evaluator import Evaluator
from src.inference import PropensityInference, RevenueInference
from src.model_loader import ModelLoader
from src.optimizer import Optimizer
from src.preprocessor import Preprocessor
from src.trainer import PropensityTrainer, RevenueTrainer

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
    inference_df = preprocessor.merge_datasets(
        datasets, base_dataset_key="Sales_Revenues"
    )
    numeric, categorical = loader.get_feature_lists()
    X_inf = inference_df[numeric + categorical]
    X_inf.index = inference_df["Client"]

    prop_inf = PropensityInference(config=config_dict)
    rev_inf = RevenueInference(config=config_dict)
    prop_preds = prop_inf.predict(X_inf)
    rev_preds = rev_inf.predict(X_inf)

    prop_inf.save(prop_preds, cfg.inference.propensity_file)
    rev_inf.save(rev_preds, cfg.inference.revenue_file)
    return prop_preds, rev_preds, X_inf.index


def run_optimization(
    cfg: DictConfig, prop_preds: pd.DataFrame, rev_preds: pd.DataFrame
) -> np.ndarray:
    """Optimize the marketing offers."""
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
    evaluator = Evaluator(
        config=config_dict, cost_per_contact=cfg.evaluation.cost_per_contact
    )
    results = evaluator.evaluate(selection, prop_preds.values, rev_preds.values)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    return results


def save_offers(
    cfg: DictConfig,
    selection: np.ndarray,
    prop_preds: pd.DataFrame,
    rev_preds: pd.DataFrame,
    client_index: pd.Index,
) -> pd.DataFrame:
    """Persist the optimized offer list to disk."""
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
                    "probability": prop_preds.loc[mask, f"probability_{product}"].values,
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
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    loader = DataLoader(config=config_dict)
    loader.config_loader.base_dir = os.path.join(get_original_cwd(), "conf")

    datasets = loader.load_configured_sheets()
    with_sales, _ = loader.create_sales_data_split(datasets)
    preprocessor = Preprocessor(loader.get_config())
    train_sets, _ = preprocessor.create_model_train_test_split(
        with_sales, test_size=cfg.preprocessing.train_test_split.test_size, random_state=cfg.preprocessing.train_test_split.random_state
    )
    merged = preprocessor.merge_datasets(train_sets, base_dataset_key="Sales_Revenues_train")
    if cfg.training.sample_fraction < 1.0:
        merged = merged.sample(frac=cfg.training.sample_fraction, random_state=cfg.training.random_seed)
    numeric, categorical = loader.get_feature_lists()
    X = merged[numeric + categorical]
    pipeline = preprocessor.create_preprocessing_pipeline(numeric, categorical)

    mlflow_cfg = loader.config.mlflow

    def train_for_product(product: str) -> None:
        y_propensity = merged[f"{cfg.targets.propensity}_{product}"]
        y_revenue = merged[f"{cfg.targets.revenue}_{product}"]
        prop_out = os.path.join(cfg.training.model_dump_path, "propensity", product)
        rev_out = os.path.join(cfg.training.model_dump_path, "revenue", product)
        os.makedirs(prop_out, exist_ok=True)
        os.makedirs(rev_out, exist_ok=True)

        if cfg.training.train_enabled:
            prop_trainer = PropensityTrainer(
                model=LogisticRegression(max_iter=200),
                preprocessor=pipeline,
                scoring=cfg.training.propensity_scoring,
                cv=cfg.training.k_folds,
                output_dir=prop_out,
                mlflow_config=mlflow_cfg,
                run_name=f"propensity-{product}",
            )
            prop_trainer.fit(X, y_propensity)

            rev_trainer = RevenueTrainer(
                model=RandomForestRegressor(),
                preprocessor=pipeline,
                scoring=cfg.training.revenue_scoring,
                cv=cfg.training.k_folds,
                output_dir=rev_out,
                mlflow_config=mlflow_cfg,
                run_name=f"revenue-{product}",
            )
            rev_trainer.fit(X, y_revenue)
        else:
            loader = ModelLoader(config=config_dict)
            loader.load_model("propensity", product)
            loader.load_model("revenue", product)
    # ---------- Load and preprocess data ----------
    with ThreadPoolExecutor(max_workers=len(cfg.products)) as executor:
        executor.map(train_for_product, cfg.products)

    # ---------- Inference ----------
    prop_preds, rev_preds, client_index = run_inference(
        cfg, preprocessor, loader, datasets, config_dict
    )

    # ---------- Optimization ----------
    selection = run_optimization(cfg, prop_preds, rev_preds)

    # ---------- Evaluation ----------
    run_evaluation(cfg, selection, prop_preds, rev_preds, config_dict)

    # ---------- Save optimized offers ----------
    save_offers(cfg, selection, prop_preds, rev_preds, client_index)


if __name__ == "__main__":
    main()
