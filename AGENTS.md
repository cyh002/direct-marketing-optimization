This Agents.md file provides comprehensive guidance for AI assistants working with this direct marketing optimization codebase.

## Project Overview

This project implements a direct marketing optimization system for financial products (Mutual Funds, Credit Cards, Consumer Loans). It analyzes customer data to predict propensity scores and optimize revenue through targeted marketing campaigns.

## Project Structure

```
direct-marketing-optimization/
├── conf/                  # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── propensity_model/  # Propensity model configs
│   └── revenue_model/     # Revenue model configs
├── data/                  # Data files
│   ├── processed/         # Processed datasets
│   └── raw/               # Raw input datasets
├── docker/                # Docker configuration
├── docs/                  # Documentation
│   └── high_propensity_list.md
├── notebooks/             # Analysis notebooks
│   ├── cc_analysis.ipynb  # Credit Card analysis
│   ├── cl_analysis.ipynb  # Consumer Loan analysis
│   ├── general_analysis.ipynb  # General dataset analysis
│   ├── mf_analysis.ipynb  # Mutual Fund analysis
│   └── prediction_analysis.ipynb
├── outputs/               # Model outputs
│   ├── models/            # Saved models
│   └── results/           # Prediction results
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── dataloader.py      # Data loading utilities
│   ├── evaluater.py       # Model evaluation
│   ├── optimizer.py       # Optimization algorithms
│   ├── preprocessor.py    # Data preprocessing
│   └── trainer.py         # Model training
├── tests/                 # Unit tests
├── main.py                # Main entry point
├── pyproject.toml         # Project dependencies
└── README.md              # Project readme
```

## Dataset Structure

The project uses several datasets for analysis:

- **Soc_Dem**: Demographic data (Client, Sex, Age, Tenure)
- **Products_ActBalance**: Product balances and counts
- **Inflow_Outflow**: Transaction patterns and volumes
- **Sales_Revenues**: Target data with sales and revenue information

## Key Components

### 1. Data Processing Components

- **dataloader.py**: Loads and validates input data
- **preprocessor.py**: Performs feature engineering and data preparation

### 2. Modeling Components

- **trainer.py**: Builds and trains propensity and revenue models
- **evaluater.py**: Evaluates model performance with metrics like AUC, precision, recall, and expected revenue

### 3. Optimization Components

- **optimizer.py**: Implements optimization algorithms for marketing campaign targeting

## Tech Stack

- **Python 3.12**: Core programming language
- **Hydra**: Configuration management with Optuna hyperparameter optimization
- **Pydantic**: Data validation and settings management
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **MLflow**: Experiment tracking and model registry
- **Pytest**: Testing framework

## Coding Conventions

### OOP Principles

This project adheres to 5 key OOP principles:
1. **Encapsulation**: Data and methods are encapsulated in classes
2. **Inheritance**: Common functionality shared through inheritance
3. **Polymorphism**: Interface consistency across different model types
4. **Abstraction**: High-level interfaces for complex operations
5. **Composition**: Building complex objects from simpler ones

### Documentation Standards

- **Docstrings**: Google style with type hints
- **Comments**: Explain complex algorithms and business logic
- **README**: High-level project overview and getting started guide

### Always update dependencies
- **Syncing dependencies**: You should run `uv sync` to sync dependencies. 
- **Adding dependencies**: run `uv add dependencies` to add your dependencies
- **Remove dependencies**: run `uv remove dependencies` to remove your dependencies 
- **Running code**: run `uv run your_code.py` to run your code. 

Example docstring format:
```python
def process_data(input_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Process raw data for model training.
    
    Args:
        input_df: Raw input dataframe
        features: List of features to process
        
    Returns:
        Processed dataframe ready for model training
        
    Raises:
        ValueError: If input_df is empty or features not found
    """
```

### Modeling Approach

- **Propensity Models**: Predict likelihood of product purchase
- **Revenue Models**: Predict expected revenue from purchase
- **Combined Optimization**: Target high propensity × high revenue customers

## Configuration and Hyperparameter Optimization

The project uses Hydra for configuration management and Optuna for hyperparameter optimization:

```python
@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the application."""
    # Access configuration
    model_config = cfg.model
    data_config = cfg.data
    
    # Rest of application logic
```

## Testing Strategy

Unit tests are organized by component:
- **Data tests**: Validate data loading and processing
- **Model tests**: Verify model training and prediction
- **Integration tests**: End-to-end workflow validation

## Experiment Tracking

MLflow is used to track experiments:
```python
with mlflow.start_run(run_name=experiment_name):
    mlflow.log_params(model_params)
    mlflow.log_metrics({"auc": auc_score, "precision": precision})
    mlflow.sklearn.log_model(model, "model")
```