"""
Data preprocessing module for machine learning pipeline.

This module provides functionality for creating train/test splits for modeling,
data merging, preprocessing pipelines, and feature engineering.
"""
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor:
    """
    Preprocessor class for data preparation and feature engineering.
    
    This class handles:
    - Traditional train/test splits for modeling
    - Data merging operations
    - Preprocessing pipeline creation
    - Feature engineering and preparation
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Initialize Preprocessor with configuration.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        
    def create_model_train_test_split(self, datasets_with_sales: Dict[str, pd.DataFrame], 
                                     test_size: float = 0.2, 
                                     random_state: int = 42) -> Tuple[Dict[str, pd.DataFrame], 
                                                                     Dict[str, pd.DataFrame]]:
        """
        Create traditional train/test split for modeling from clients with sales data.
        
        This splits the clients who have sales data into train/test sets for 
        model development and evaluation.
        
        Parameters
        ----------
        datasets_with_sales : Dict[str, pd.DataFrame]
            Dictionary of datasets containing only clients with sales data
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            (datasets_train, datasets_test) dictionaries for modeling
            
        Raises
        ------
        ValueError
            If no Sales_Revenues dataset found in datasets_with_sales
        """
        # Get unique clients from sales data
        sales_dataset_key = next(
            (k for k in datasets_with_sales.keys() if 'Sales_Revenues' in k), None
        )
        if not sales_dataset_key:
            raise ValueError("No Sales_Revenues dataset found in datasets_with_sales")
        
        all_clients = datasets_with_sales[sales_dataset_key]['Client'].unique()
        train_clients, test_clients = train_test_split(
            all_clients, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Model split: {len(train_clients)} train clients, {len(test_clients)} test clients")
        
        datasets_train = {}
        datasets_test = {}
        
        for name, df in datasets_with_sales.items():
            if 'Client' in df.columns:
                train_df = df[df['Client'].isin(train_clients)].copy()
                test_df = df[df['Client'].isin(test_clients)].copy()
                
                # Remove '_with_sales' suffix and add train/test suffix
                base_name = name.replace('_with_sales', '')
                datasets_train[f"{base_name}_train"] = train_df
                datasets_test[f"{base_name}_test"] = test_df
        
        return datasets_train, datasets_test
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame],
                      base_dataset_key: str,
                      join_key: str = 'Client',
                      datasets_to_merge: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge multiple datasets on a specified key.
        
        Parameters
        ----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary containing all datasets to merge
        base_dataset_key : str
            Key for the base dataset
        join_key : str, default='Client'
            Column name to join on
        datasets_to_merge : List[str], optional
            List of dataset keys to merge. If None, merges all except base
            
        Returns
        -------
        pd.DataFrame
            Merged dataset
            
        Raises
        ------
        ValueError
            If base dataset or join key is not found
        """
        print("=== MERGING DATASETS ===")
        
        # Validate base dataset exists
        if base_dataset_key not in datasets:
            raise ValueError(f"Base dataset '{base_dataset_key}' not found in datasets")
        
        # Start with base dataset
        merged_df = datasets[base_dataset_key].copy()
        print(f"Base dataset ({base_dataset_key}): {merged_df.shape}")
        
        # Determine datasets to merge
        if datasets_to_merge is None:
            datasets_to_merge = [key for key in datasets.keys() if key != base_dataset_key]
        
        # Validate join key exists in base dataset
        if join_key not in merged_df.columns:
            raise ValueError(f"Join key '{join_key}' not found in base dataset")
        
        # Merge each dataset
        for dataset_name in datasets_to_merge:
            if dataset_name in datasets:
                merged_df = self._merge_single_dataset(
                    merged_df, datasets[dataset_name], dataset_name, join_key
                )
            else:
                print(f"  ⚠️  {dataset_name} not found in datasets")
        
        print(f"\n=== FINAL MERGED DATASET ===")
        print(f"Shape: {merged_df.shape}")
        print(f"Unique {join_key}s: {merged_df[join_key].nunique()}")
        print(f"Columns: {len(merged_df.columns)}")
        
        return merged_df
    
    def _merge_single_dataset(self, merged_df: pd.DataFrame, 
                             df_to_merge: pd.DataFrame,
                             dataset_name: str, 
                             join_key: str) -> pd.DataFrame:
        """
        Merge a single dataset into the main merged dataset.
        
        Parameters
        ----------
        merged_df : pd.DataFrame
            Current merged dataset
        df_to_merge : pd.DataFrame
            Dataset to merge in
        dataset_name : str
            Name of the dataset being merged
        join_key : str
            Column to join on
            
        Returns
        -------
        pd.DataFrame
            Updated merged dataset
        """
        print(f"\nMerging {dataset_name}: {df_to_merge.shape}")
        
        # Check for join key column
        if join_key in df_to_merge.columns:
            before_shape = merged_df.shape
            merged_df = merged_df.merge(
                df_to_merge, on=join_key, how='left', suffixes=('', '_dup')
            )
            after_shape = merged_df.shape
            
            print(f"  Before merge: {before_shape}")
            print(f"  After merge: {after_shape}")
            print(f"  Records with {join_key}: {merged_df[join_key].nunique()}")
            
            # Handle duplicate columns
            duplicate_cols = [col for col in merged_df.columns if col.endswith('_dup')]
            if duplicate_cols:
                original_cols = [col.replace('_dup', '') for col in duplicate_cols]
                print(f"  ⚠️  Duplicate columns found: {original_cols}")
                merged_df = merged_df.drop(columns=duplicate_cols)
                print("  ✅ Dropped duplicate columns")
                
        else:
            print(f"  ⚠️  No '{join_key}' column found in {dataset_name}")
        
        return merged_df
    
    def create_preprocessing_pipeline(self, numeric_features: List[str],
                                    categorical_features: List[str]) -> ColumnTransformer:
        """
        Create preprocessing pipeline based on configuration.
        
        Parameters
        ----------
        numeric_features : List[str]
            List of numeric feature column names
        categorical_features : List[str]
            List of categorical feature column names
            
        Returns
        -------
        ColumnTransformer
            Configured preprocessing pipeline
        """
        # Get preprocessing config
        preprocess_config = self.config.get('preprocessing', {})
        
        # Numeric transformer
        numeric_imputer_config = preprocess_config.get('numeric_imputer', {})
        numeric_transformer = SimpleImputer(
            strategy=numeric_imputer_config.get('strategy', 'constant'),
            fill_value=numeric_imputer_config.get('fill_value', 0)
        )
        
        # Categorical transformer
        categorical_imputer_config = preprocess_config.get('categorical_imputer', {})
        onehot_config = preprocess_config.get('onehot', {})
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(
                strategy=categorical_imputer_config.get('strategy', 'constant'),
                fill_value=categorical_imputer_config.get('fill_value', 'missing')
            )),
            ('onehot', OneHotEncoder(
                handle_unknown=onehot_config.get('handle_unknown', 'ignore')
            ))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])
        
        return preprocessor
    
    def analyze_data_quality(self, df: pd.DataFrame, 
                           show_columns: bool = True,
                           max_missing_display: int = 20) -> None:
        """
        Analyze data quality of a dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze
        show_columns : bool, default=True
            Whether to display all column names
        max_missing_display : int, default=20
            Maximum number of missing value columns to display
        """
        print("=== DATA QUALITY CHECK ===")
        print(f"Dataset shape: {df.shape}")
        
        # Display column names if requested
        if show_columns:
            print("\nColumn names:")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:2d}. {col}")
        
        # Check for missing values
        self._analyze_missing_values(df, max_missing_display)
        
        # Check data types
        self._analyze_data_types(df)
        
        # Check for potential issues
        self._analyze_data_issues(df)
    
    def _analyze_missing_values(self, df: pd.DataFrame, 
                               max_missing_display: int) -> None:
        """Analyze missing values in the dataset."""
        print("\nMissing values per column:")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            print(f"Found {len(missing_data)} columns with missing values:")
            display_missing = missing_data.head(max_missing_display)
            for col, count in display_missing.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
            
            if len(missing_data) > max_missing_display:
                remaining = len(missing_data) - max_missing_display
                print(f"  ... and {remaining} more columns")
        else:
            print("No missing values found! ✅")
    
    def _analyze_data_types(self, df: pd.DataFrame) -> None:
        """Analyze data types in the dataset."""
        print("\nData types summary:")
        dtype_summary = df.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")
    
    def _analyze_data_issues(self, df: pd.DataFrame) -> None:
        """Analyze potential data issues."""
        print("\nPotential issues:")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠️  {duplicates} duplicate rows found")
        else:
            print("  ✅ No duplicate rows")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"  ⚠️  Constant columns (nunique ≤ 1): {constant_cols}")
        else:
            print("  ✅ No constant columns")


# Convenience functions
def load_and_create_model_splits(config_path: str, 
                               test_size: float = 0.2) -> Tuple[Dict[str, pd.DataFrame], 
                                                               Dict[str, pd.DataFrame], 
                                                               Dict[str, pd.DataFrame], 
                                                               Any]:
    """
    Convenience function to load datasets and create both splits.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    test_size : float, default=0.2
        Proportion for model test split
    
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Any]
        (datasets_train, datasets_test, datasets_without_sales, loader)
    """
    from .dataloader import DataLoader  # Local import to avoid circular dependency
    
    loader = DataLoader(config_path=config_path)
    datasets = loader.load_configured_sheets()
    datasets_with_sales, datasets_without_sales = loader.create_sales_data_split(datasets)
    
    preprocessor = Preprocessor(loader.get_config())
    datasets_train, datasets_test = preprocessor.create_model_train_test_split(
        datasets_with_sales, test_size
    )
    
    return datasets_train, datasets_test, datasets_without_sales, loader