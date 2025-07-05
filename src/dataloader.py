"""
Data loading module for Excel dataset handling.

This module provides functionality for loading Excel datasets and creating
business-logic splits based on sales data availability.
"""
import os
from typing import Dict, Optional, List, Tuple
import pandas as pd
import yaml


class DataLoader:
    """
    Data loader class for handling Excel dataset loading and processing.
    
    This class handles:
    - Loading Excel files with multiple sheets
    - Filtering to configured sheets only
    - Creating business logic splits (clients with/without sales data)
    - Providing access to configuration data
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 config: Optional[Dict] = None) -> None:
        """
        Initialize DataLoader with configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration YAML file
        config : dict, optional
            Configuration dictionary (alternative to config_path)
            
        Raises
        ------
        ValueError
            If neither config_path nor config is provided
        FileNotFoundError
            If config_path is provided but file doesn't exist
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
    
    def load_excel_datasets(self, excel_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load datasets from Excel file.
        
        Parameters
        ----------
        excel_path : str, optional
            Path to Excel file. If None, uses path from config
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing datasets from Excel sheets
            
        Raises
        ------
        FileNotFoundError
            If Excel file is not found
        Exception
            If there's an error loading the Excel file
        """
        # Use provided path or get from config
        if excel_path is None:
            excel_path = self.config['data']['raw_excel_path']
        
        # Handle relative paths
        if not os.path.isabs(excel_path):
            current_dir = os.getcwd()
            excel_path = os.path.join(current_dir, excel_path)
        
        print(f"Loading Excel file from: {excel_path}")
        
        try:
            # Load Excel file
            xls = pd.ExcelFile(excel_path, engine="openpyxl")
            print(f"All sheet names: {list(xls.sheet_names)}")
            
            # Load all sheets as datasets
            datasets = {}
            for sheet in xls.sheet_names:
                datasets[sheet] = pd.read_excel(
                    xls, sheet_name=sheet, engine="openpyxl"
                )
            
            print(f"Available datasets: {list(datasets.keys())}")
            
            # Remove Description sheet if it exists
            if 'Description' in datasets:
                datasets.pop('Description')
                print("Removed 'Description' sheet from datasets")
            
            # Print dataset summary
            for name, df in datasets.items():
                print(f"{name}: {len(df)} rows, {len(df.columns)} columns")
            
            return datasets
            
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Excel file not found at: {excel_path}") from exc
        except Exception as exc:
            raise Exception(f"Error loading Excel file: {str(exc)}") from exc
    
    def load_configured_sheets(self, excel_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load only the sheets specified in configuration.
        
        Parameters
        ----------
        excel_path : str, optional
            Path to Excel file. If None, uses path from config
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing only configured datasets
        """
        # Get all datasets first
        all_datasets = self.load_excel_datasets(excel_path)
        
        # Filter to only configured sheets
        configured_sheets = self.config['data']['sheets']
        configured_datasets = {}
        
        for sheet in configured_sheets:
            if sheet in all_datasets:
                configured_datasets[sheet] = all_datasets[sheet]
                print(f"✅ Loaded configured sheet: {sheet}")
            else:
                print(f"⚠️  Configured sheet '{sheet}' not found in Excel file")
        
        return configured_datasets
    
    def create_sales_data_split(self, datasets: Dict[str, pd.DataFrame], 
                               sales_key: str = 'Sales_Revenues') -> Tuple[Dict[str, pd.DataFrame], 
                                                                         Dict[str, pd.DataFrame]]:
        """
        Split datasets based on Sales_Revenues data availability.
        
        This creates a business logic split based on which clients have
        sales/revenue history, NOT a traditional train/test split for modeling.
        
        Parameters
        ----------
        datasets : Dict[str, pd.DataFrame]
            Dictionary of datasets
        sales_key : str, default='Sales_Revenues'
            Key for the sales/revenues dataset
            
        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            (datasets_with_sales, datasets_without_sales) dictionaries
            
        Raises
        ------
        ValueError
            If sales dataset is not found in datasets
        """
        if sales_key not in datasets:
            raise ValueError(f"Sales dataset '{sales_key}' not found in datasets")
        
        # Get clients from Sales_Revenues dataset
        sales_clients = set(datasets[sales_key]['Client'].unique())
        print(f"Clients with sales data in {sales_key}: {len(sales_clients)}")
        
        # Create splits based on sales data availability
        datasets_with_sales = {}
        datasets_without_sales = {}
        
        for name, df in datasets.items():
            if 'Client' in df.columns:
                # With sales: clients that exist in Sales_Revenues
                with_sales_mask = df['Client'].isin(sales_clients)
                with_sales_df = df[with_sales_mask].copy()
                
                # Without sales: clients that do NOT exist in Sales_Revenues
                without_sales_df = df[~with_sales_mask].copy()
                
                # Store with descriptive names
                datasets_with_sales[f"{name}_with_sales"] = with_sales_df
                datasets_without_sales[f"{name}_without_sales"] = without_sales_df
                
                self._print_split_summary(name, df, with_sales_df, without_sales_df)
        
        return datasets_with_sales, datasets_without_sales
    
    def _print_split_summary(self, name: str, original_df: pd.DataFrame,
                           with_sales_df: pd.DataFrame, 
                           without_sales_df: pd.DataFrame) -> None:
        """
        Print summary of split operation.
        
        Parameters
        ----------
        name : str
            Name of the dataset
        original_df : pd.DataFrame
            Original dataset
        with_sales_df : pd.DataFrame
            Dataset with sales data
        without_sales_df : pd.DataFrame
            Dataset without sales data
        """
        total_rows = len(original_df)
        with_sales_rows = len(with_sales_df)
        without_sales_rows = len(without_sales_df)
        
        print(f"{name}:")
        print(f"  Total rows: {total_rows}")
        print(f"  With sales data: {with_sales_rows} "
              f"({with_sales_rows/total_rows*100:.1f}%)")
        print(f"  Without sales data: {without_sales_rows} "
              f"({without_sales_rows/total_rows*100:.1f}%)")
        print(f"  Unique clients with sales: {with_sales_df['Client'].nunique()}")
        print(f"  Unique clients without sales: {without_sales_df['Client'].nunique()}")
        print()
    
    def get_feature_lists(self) -> Tuple[List[str], List[str]]:
        """
        Get feature lists from configuration.
        
        Returns
        -------
        Tuple[List[str], List[str]]
            (numeric_features, categorical_features)
        """
        numeric_features = self.config['features']['numeric']
        categorical_features = self.config['features']['categorical']
        
        return numeric_features, categorical_features
    
    def get_products(self) -> List[str]:
        """
        Get list of products from configuration.
        
        Returns
        -------
        List[str]
            List of product codes
        """
        return self.config['products']
    
    def get_targets(self) -> Dict[str, str]:
        """
        Get target configuration.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping target types to column names
        """
        return self.config['targets']
    
    def get_config(self) -> Dict:
        """
        Get the full configuration dictionary.
        
        Returns
        -------
        Dict
            Complete configuration dictionary
        """
        return self.config


# Convenience functions
def load_datasets_from_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load datasets using configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing datasets
    """
    loader = DataLoader(config_path=config_path)
    return loader.load_configured_sheets()


def load_and_create_sales_split(config_path: str) -> Tuple[Dict[str, pd.DataFrame], 
                                                          Dict[str, pd.DataFrame], 
                                                          DataLoader]:
    """
    Convenience function to load datasets and split by sales data availability.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], DataLoader]
        (datasets_with_sales, datasets_without_sales, loader)
    """
    loader = DataLoader(config_path=config_path)
    datasets = loader.load_configured_sheets()
    datasets_with_sales, datasets_without_sales = loader.create_sales_data_split(datasets)
    
    return datasets_with_sales, datasets_without_sales, loader