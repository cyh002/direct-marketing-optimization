This Agents.md file provides comprehensive guidance for AI assistants working with this direct marketing optimization codebase.

## Project Overview

This project implements a direct marketing optimization system for financial products (Mutual Funds, Credit Cards, Consumer Loans). It analyzes customer data to predict propensity scores and optimize revenue through targeted marketing campaigns.

## Project Structure

```
.
├── conf/                 # Hydra configuration files
├── data/                 # Input data (raw/processed)
├── docker/               # Dockerfile for the app
├── docker-compose.yml    # Compose file to run MLflow and the app
├── docs/                 # Additional documentation
├── notebooks/            # Exploratory notebooks
├── src/                  # Source code package
├── tests/                # Unit tests
└── main.py               # Entry point for the workflow
```

## Dataset Structure

The project uses several datasets for analysis:

- **Soc_Dem**: Demographic data (Client, Sex, Age, Tenure)
- **Products_ActBalance**: Product balances and counts
- **Inflow_Outflow**: Transaction patterns and volumes
- **Sales_Revenues**: Target data with sales and revenue information

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
