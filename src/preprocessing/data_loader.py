"""
Data Loader Module

This module provides the DataLoader class for loading and merging
multiple CSV files from the coffee project dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import RAW_DATA_DIR, MERGED_DATA_FILE, CLEANED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for loading and merging multiple CSV datasets.
    
    This class handles reading CSV files from the data/raw directory
    and merging them based on ID, City, and Year columns.
    """
    
    def __init__(self, raw_data_dir: Optional[str] = None, use_cleaned: bool = False):
        """
        Initialize DataLoader with the path to raw data directory.
        
        Args:
            raw_data_dir: Path to the directory containing raw CSV files.
                         If None, uses the path from config.py
            use_cleaned: If True, load from cleaned data directory instead of raw
        """
        self.use_cleaned = use_cleaned
        
        if raw_data_dir is None:
            if use_cleaned:
                self.raw_data_dir = Path(CLEANED_DATA_DIR)
            else:
                self.raw_data_dir = Path(RAW_DATA_DIR)
        else:
            self.raw_data_dir = Path(raw_data_dir)
            
        self.data: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None
        
        # Define file names and their separators
        if use_cleaned:
            self.file_names = {
                'sa_var': 'sa_var_clean.csv',
                'needstate': 'needstate_clean.csv',
                'brandhealth': 'brandhealth_clean.csv',
                'brand_image': 'brand_image_clean.csv',
                'segmentation': 'customer_segmentation_clean.csv'
            }
            # Cleaned files all use comma separator
            self.file_separators = {
                'sa_var': ',',
                'needstate': ',',
                'brandhealth': ',',
                'brand_image': ',',
                'segmentation': ','
            }
        else:
            self.file_names = {
                'sa_var': 'SA#var.csv',
                'needstate': 'NeedstateDayDaypart.csv',
                'brandhealth': 'Brandhealth.csv',
                'brand_image': 'Brand_Image.csv',
                'segmentation': '2017Segmentation3685case.csv'
            }
            # Define separator for each file
            self.file_separators = {
                'sa_var': ';',
                'needstate': ';',
                'brandhealth': ',',
                'brand_image': ',',
                'segmentation': ';'
            }
    
    def _read_csv(self, file_name: str, key: str) -> pd.DataFrame:
        """
        Read a single CSV file.
        
        Args:
            file_name: Name of the CSV file to read
            key: Key to store the DataFrame in self.data dict
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        file_path = self.raw_data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading {file_name}...")
        
        # Get the correct separator for this file
        sep = self.file_separators.get(key, ';')
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                logger.info(f"Loaded {file_name}: shape = {df.shape}, sep='{sep}', encoding={encoding}")
                self.data[key] = df
                return df
            except:
                continue
        
        # If all attempts failed, raise error
        raise ValueError(f"Could not read {file_name} with separator '{sep}' and any encoding")

        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the raw data directory.
        
        Returns:
            Dictionary containing all loaded DataFrames
            
        Raises:
            FileNotFoundError: If any required file is missing
        """
        logger.info("Starting to load all CSV files...")
        
        for key, file_name in self.file_names.items():
            self._read_csv(file_name, key)
        
        logger.info(f"Successfully loaded {len(self.data)} files")
        return self.data
    
    def merge_all(self) -> pd.DataFrame:
        """
        Merge all loaded DataFrames into a single DataFrame.
        
        Merging strategy:
        1. Start with SA_var
        2. Left join with NeedstateDaydaypart on [ID, City, Year]
        3. Left join with Brandhealth on [ID, City, Year]
        4. Left join with Brand_Image on [ID, City, Year]
        5. Left join with Segmentation on [ID] (selected columns only)
        
        Returns:
            Merged DataFrame
            
        Raises:
            ValueError: If required columns for merging are missing
        """
        # Load data if not already loaded
        if not self.data:
            self.load_all()
        
        logger.info("Starting merge process...")
        
        # Filter chỉ lấy năm 2017 (vì các bảng brandhealth, needstate, brand_image chỉ có data 2017)
        if 'Year' in self.data['sa_var'].columns:
            sa_var_2017 = self.data['sa_var'][self.data['sa_var']['Year'] == 2017].copy()
            logger.info(f"Filtered SA_var to year 2017: {len(self.data['sa_var'])} -> {len(sa_var_2017)} rows")
        else:
            sa_var_2017 = self.data['sa_var'].copy()
        
        # Verify required merge columns exist
        merge_cols = ['ID', 'City', 'Year']
        for key in ['sa_var', 'needstate', 'brandhealth', 'brand_image']:
            df = self.data[key]
            missing_cols = [col for col in merge_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{key} is missing columns: {missing_cols}")
        
        # Remove duplicates from each dataset before merging
        logger.info("Removing duplicates from datasets...")
        needstate_dedup = self.data['needstate'].drop_duplicates(subset=['ID', 'City', 'Year'], keep='first')
        brandhealth_dedup = self.data['brandhealth'].drop_duplicates(subset=['ID', 'City', 'Year'], keep='first')
        
        # Xóa cột PPA và Spending từ brandhealth vì đã có PPA_seg, Spending_seg từ segmentation
        cols_to_drop = ['PPA', 'Spending']
        for col in cols_to_drop:
            if col in brandhealth_dedup.columns:
                brandhealth_dedup = brandhealth_dedup.drop(columns=[col])
                logger.info(f"Dropped column '{col}' from brandhealth (will use from segmentation)")
        
        brand_image_dedup = self.data['brand_image'].drop_duplicates(subset=['ID', 'City', 'Year'], keep='first')
        
        logger.info(f"NeedstateDaydaypart: {len(self.data['needstate'])} -> {len(needstate_dedup)} rows")
        logger.info(f"Brandhealth: {len(self.data['brandhealth'])} -> {len(brandhealth_dedup)} rows")
        logger.info(f"Brand_Image: {len(self.data['brand_image'])} -> {len(brand_image_dedup)} rows")
        
        # Start with SA_var 2017 as base
        merged = sa_var_2017.copy()
        logger.info(f"Base (SA_var 2017): shape = {merged.shape}")
        
        # Merge with NeedstateDaydaypart
        merged = merged.merge(
            needstate_dedup,
            on=['ID', 'City', 'Year'],
            how='left',  # Chỉ giữ ID có trong cả 2 bảng
            suffixes=('', '_ns')
        )
        logger.info(f"After merging NeedstateDaydaypart: shape = {merged.shape}")
        
        # Merge with Brandhealth
        merged = merged.merge(
            brandhealth_dedup,
            on=['ID', 'City', 'Year'],
            how='left',  # Chỉ giữ ID có trong cả 2 bảng
            suffixes=('', '_bh')
        )
        logger.info(f"After merging Brandhealth: shape = {merged.shape}")
        
        # Merge with Brand_Image
        merged = merged.merge(
            brand_image_dedup,
            on=['ID', 'City', 'Year'],
            how='left',  # Chỉ giữ ID có trong cả 2 bảng
            suffixes=('', '_bi')
        )
        logger.info(f"After merging Brand_Image: shape = {merged.shape}")
        
        # Merge with Segmentation (selected columns only)
        seg_cols = ['ID', 'Visit', 'Spending', 'PPA']
        
        # Verify segmentation columns exist
        missing_seg_cols = [col for col in seg_cols if col not in self.data['segmentation'].columns]
        if missing_seg_cols:
            raise ValueError(f"Segmentation file is missing columns: {missing_seg_cols}")
        
        segmentation_subset = self.data['segmentation'][seg_cols].copy()
        
        merged = merged.merge(
            segmentation_subset,
            on='ID',
            how='left',
            suffixes=('', '_seg')
        )
        logger.info(f"After merging Segmentation: shape = {merged.shape}")
        
        # Xóa cột Awareness từ brandhealth, giữ Awareness_bi từ brand_image
        if 'Awareness' in merged.columns:
            merged = merged.drop(columns=['Awareness'])
            logger.info("Dropped 'Awareness' column from brandhealth")
        
        # Xóa cột Frequency_Visit
        if 'Frequency_Visit' in merged.columns:
            merged = merged.drop(columns=['Frequency_Visit'])
            logger.info("Dropped 'Frequency_Visit' column")
        
        # Đổi tên cột từ _seg và _bi về tên gốc
        rename_map = {
            'Visit_seg': 'Visit',
            'Spending_seg': 'Spending',
            'PPA_seg': 'PPA',
            'Awareness_bi': 'Awareness'
        }
        merged = merged.rename(columns=rename_map)
        
        # Fill các cột categorical NaN sau merge
        categorical_cols_to_fill = [
            'Spontaneous', 'Trial', 'P3M', 'P1M', 'Comprehension',
            'Brand_Likability', 'Weekly', 'Daily', 'Awareness', 'Attribute', 
            'BrandImage', 'NPS_P3M_Group', 'Segmentation', 'Needstates', 
            'Daypart', 'NeedstateGroup', 'Brand', 'IncomeBand'
        ]
        
        for col in categorical_cols_to_fill:
            if col in merged.columns:
                filled_count = merged[col].isna().sum()
                if filled_count > 0:
                    merged[col] = merged[col].fillna('Unknown')
                    logger.info(f"Filled {filled_count} NaN values in '{col}' with 'Unknown'")
        
        # Xóa cột IncomeBand vì toàn Unknown cho năm 2017
        if 'IncomeBand' in merged.columns:
            merged = merged.drop(columns=['IncomeBand'])
            logger.info("Dropped 'IncomeBand' column (all values are Unknown for year 2017)")
        
        # Fill các cột numerical NaN sau merge
        numerical_cols_to_fill = [
            'Visit', 'Spending', 'PPA', 'NPS_P3M', 'Group_size', 'Age'
        ]
        
        for col in numerical_cols_to_fill:
            if col in merged.columns:
                filled_count = merged[col].isna().sum()
                if filled_count > 0:
                    merged[col] = merged[col].fillna(-1)
                    logger.info(f"Filled {filled_count} NaN values in '{col}' with -1")
        
        self.merged_df = merged
        logger.info(f"Merge complete. Final shape: {merged.shape}")
        
        return merged
    
    def save_merged(self, out_path: Optional[str] = None) -> None:
        """
        Save the merged DataFrame to a CSV file.
        
        Args:
            out_path: Output path for the merged CSV file.
                     If None, uses the path from config.py
            
        Raises:
            ValueError: If merge_all() hasn't been called yet
        """
        if self.merged_df is None:
            raise ValueError("No merged data available. Call merge_all() first.")
        
        if out_path is None:
            output_path = Path(MERGED_DATA_FILE)
        else:
            output_path = Path(out_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged data to {output_path}...")
        self.merged_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved merged data to {output_path}")
    
    def get_merged_info(self) -> Tuple[int, int]:
        """
        Get information about the merged DataFrame.
        
        Returns:
            Tuple of (number of rows, number of columns)
            
        Raises:
            ValueError: If merge_all() hasn't been called yet
        """
        if self.merged_df is None:
            raise ValueError("No merged data available. Call merge_all() first.")
        
        return self.merged_df.shape


if __name__ == "__main__":
    # Test the DataLoader with cleaned data
    dl = DataLoader(use_cleaned=True)
    df = dl.merge_all()
    print(f"Merged DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    dl.save_merged()
