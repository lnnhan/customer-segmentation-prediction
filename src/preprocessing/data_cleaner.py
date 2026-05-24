"""
Module làm sạch dữ liệu

Module này cung cấp class DataCleaner để làm sạch và tiền xử lý
các file CSV từ dự án phân cụm khách hàng F&B.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Class DataCleaner để làm sạch và tiền xử lý nhiều datasets.
    
    Class này xử lý các thao tác làm sạch dữ liệu cho customer segmentation,
    brand image, brand health, SA variables và needstate datasets.
    """
    
    def __init__(self):
        """Khởi tạo DataCleaner."""
        pass
    
    def _normalize_string_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Chuẩn hóa các cột string: strip khoảng trắng.
        
        Args:
            df: DataFrame
            cols: Danh sách tên cột cần chuẩn hóa
            
        Returns:
            DataFrame đã chuẩn hóa
        """
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df
    
    def _winsorize_series(self, s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
        """
        Áp dụng Winsorization (capping) cho một Series.
        
        Args:
            s: Series cần xử lý
            lower_q: Quantile dưới (mặc định 0.01)
            upper_q: Quantile trên (mặc định 0.99)
            
        Returns:
            Series đã được winsorize
        """
        lower_bound = s.quantile(lower_q)
        upper_bound = s.quantile(upper_q)
        return s.clip(lower=lower_bound, upper=upper_bound)
    
    def _map_income_band_from_mean(self, x: float) -> Optional[str]:
        """
        Ánh xạ giá trị thu nhập trung bình sang nhóm thu nhập.
        
        Args:
            x: Giá trị thu nhập trung bình
            
        Returns:
            Nhóm thu nhập tương ứng
        """
        if pd.isna(x):
            return None
        if 0 <= x < 3000:
            return "Under 3 millions VND"
        if 3000 <= x < 4500:
            return "From 3 millions to 4.49 millions VND"
        if 4500 <= x < 6500:
            return "From 4.5 millions to 6.49 millions VND"
        if 6500 <= x < 7500:
            return "From 6.5 millions to 7.49 millions VND"
        if 7500 <= x < 9000:
            return "From 7.5 millions to 8.99 millions VND"
        if 9000 <= x < 12000:
            return "From 9 millions to 11.99 millions VND"
        if 12000 <= x < 15000:
            return "From 12 millions to 14.99 millions VND"
        if 15000 <= x < 20000:
            return "From 15 millions to 19.99 millions VND"
        if 20000 <= x < 25000:
            return "From 20 millions to 24.99 millions VND"
        if 25000 <= x < 30000:
            return "From 25 millions to 29.99 millions VND"
        if 30000 <= x < 45000:
            return "From 30 millions to 44.99 millions VND"
        if 45000 <= x < 75000:
            return "From 45 millions to 74.99 millions VND"
        if x >= 75000:
            return "From 75 million to VND 149.99 million VND"
        return None
    
    def _fix_brand_spelling(self, value: str) -> str:
        """
        Sửa lỗi chính tả trong tên brand.
        
        Args:
            value: Giá trị brand
            
        Returns:
            Giá trị đã sửa
        """
        if pd.isna(value):
            return value
        
        spelling_map = {
            'Indepentdent': 'Independent',
            'Indepedent Cafe': 'Independent Cafe',
            'Indepentdent Cafe': 'Independent Cafe'
        }
        
        return spelling_map.get(value, value)
    
    def clean_customer_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu customer segmentation.
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        logger.info("Đang làm sạch dữ liệu customer segmentation...")
        
        # Loại bỏ duplicate theo ID
        df = df.drop_duplicates(subset=['ID'], keep='first')
        
        # Chuyển ID sang object
        df['ID'] = df['ID'].astype(str)
        
        # Chuẩn hóa text
        text_cols = ['Brand', 'Segmentation']
        df = self._normalize_string_columns(df, text_cols)
        
        # Sửa lỗi chính tả Brand
        if 'Brand' in df.columns:
            df['Brand'] = df['Brand'].apply(self._fix_brand_spelling)
        
        logger.info(f"Đã làm sạch customer segmentation: shape = {df.shape}")
        return df
    
    def clean_brand_image(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu brand image.
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        logger.info("Đang làm sạch dữ liệu brand image...")
        
        # Loại bỏ duplicate rows
        df = df.drop_duplicates(keep='first')
        
        # Chuyển ID sang object
        df['ID'] = df['ID'].astype(str)
        
        # Chuẩn hóa text
        text_cols = [col for col in df.select_dtypes(include=['object']).columns]
        df = self._normalize_string_columns(df, text_cols)
        
        # Fill Awareness NaN bằng "Highlands Coffee"
        if 'Awareness' in df.columns:
            df['Awareness'] = df['Awareness'].fillna('Highlands Coffee')
        
        logger.info(f"Đã làm sạch brand image: shape = {df.shape}")
        return df
    
    def clean_brandhealth(
        self, 
        df: pd.DataFrame,
        brand_image_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Làm sạch dữ liệu brandhealth.
        
        Args:
            df: DataFrame đầu vào
            brand_image_df: DataFrame brand image tùy chọn để điền Brand_Likability
            
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        logger.info("Đang làm sạch dữ liệu brandhealth...")
        
        # Loại bỏ duplicate
        df = df.drop_duplicates(keep='first')
        
        # Chuyển ID sang object
        df['ID'] = df['ID'].astype(str)
        
        # Xóa các dòng có Awareness NaN
        if 'Awareness' in df.columns:
            df = df.dropna(subset=['Awareness'])
        
        # Đổi tên cột
        rename_map = {
            'Fre#visit': 'Frequency_Visit',
            'NPS#P3M': 'NPS_P3M',
            'NPS#P3M#Group': 'NPS_P3M_Group'
        }
        df = df.rename(columns=rename_map)
        
        # Xử lý Spontaneous
        if 'Spontaneous' in df.columns and 'Awareness' in df.columns:
            mask = df['Spontaneous'].isna() & df['Awareness'].notna()
            df.loc[mask, 'Spontaneous'] = df.loc[mask, 'Awareness']
            df['Spontaneous'] = df['Spontaneous'].fillna('Unknown')
        
        # Xử lý Trial
        if 'Trial' in df.columns:
            highland_cols = ['P3M', 'P1M', 'Spontaneous', 'Brand_Likability', 'Weekly', 'Daily']
            mask_fill = df['Trial'].isna()
            for col in highland_cols:
                if col in df.columns:
                    mask_fill = mask_fill & (df[col] == 'Highlands Coffee')
            df.loc[mask_fill, 'Trial'] = 'Highlands Coffee'
            df['Trial'] = df['Trial'].fillna('Unknown')
        
        # Xử lý P3M
        if 'P3M' in df.columns:
            if 'P1M' in df.columns:
                mask = df['P3M'].isna() & df['P1M'].notna()
                df.loc[mask, 'P3M'] = df.loc[mask, 'P1M']
            
            if 'Daily' in df.columns:
                mask = df['P3M'].isna() & df['Daily'].notna()
                df.loc[mask, 'P3M'] = df.loc[mask, 'Daily']
            
            if 'Weekly' in df.columns:
                mask = df['P3M'].isna() & df['Weekly'].notna()
                df.loc[mask, 'P3M'] = df.loc[mask, 'Weekly']
            
            df['P3M'] = df['P3M'].fillna('Unknown')
        
        # Xử lý P1M
        if 'P1M' in df.columns:
            if 'Daily' in df.columns:
                mask = df['P1M'].isna() & df['Daily'].notna()
                df.loc[mask, 'P1M'] = df.loc[mask, 'Daily']
            
            if 'Weekly' in df.columns:
                mask = df['P1M'].isna() & df['Weekly'].notna()
                df.loc[mask, 'P1M'] = df.loc[mask, 'Weekly']
            
            df['P1M'] = df['P1M'].fillna('Unknown')
        
        # Xử lý Comprehension
        if 'Comprehension' in df.columns and 'Awareness' in df.columns:
            mask = (df['Awareness'] == 'Highlands Coffee') & df['Comprehension'].isna()
            df.loc[mask, 'Comprehension'] = 'Know a little'
            df['Comprehension'] = df['Comprehension'].fillna('Unknown')
        
        # Xử lý Frequency_Visit
        if 'Frequency_Visit' in df.columns:
            fre_median = df['Frequency_Visit'].median()
            highland_cols = ['Trial', 'P3M', 'P1M', 'Brand_Likability', 'Weekly', 'Daily']
            
            mask_highland = pd.Series([False] * len(df), index=df.index)
            for col in highland_cols:
                if col in df.columns:
                    mask_highland = mask_highland | (df[col] == 'Highlands Coffee')
            
            mask_fill_median = df['Frequency_Visit'].isna() & mask_highland
            mask_fill_zero = df['Frequency_Visit'].isna() & ~mask_highland
            
            df.loc[mask_fill_median, 'Frequency_Visit'] = fre_median
            df.loc[mask_fill_zero, 'Frequency_Visit'] = 0
        
        # Xử lý Brand_Likability
        if 'Brand_Likability' in df.columns and 'Frequency_Visit' in df.columns:
            mask = df['Brand_Likability'].isna() & (df['Frequency_Visit'] > 3)
            if 'Brand' in df.columns:
                df.loc[mask, 'Brand_Likability'] = df.loc[mask, 'Brand']
            
            # Fill từ brand_image nếu có
            if brand_image_df is not None and 'BrandImage' in brand_image_df.columns:
                brand_image_df = brand_image_df.copy()
                brand_image_df['ID'] = brand_image_df['ID'].astype(str)
                brand_image_map = brand_image_df.set_index('ID')['BrandImage'].to_dict()
                mask = df['Brand_Likability'].isna()
                df.loc[mask, 'Brand_Likability'] = df.loc[mask, 'ID'].map(brand_image_map)
            
            df['Brand_Likability'] = df['Brand_Likability'].fillna('Unknown')
        
        # Xử lý Weekly và Daily
        for col in ['Weekly', 'Daily']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Xóa Spending_use nếu giống Spending
        if 'Spending' in df.columns and 'Spending_use' in df.columns:
            if df['Spending'].equals(df['Spending_use']):
                df = df.drop(columns=['Spending_use'])
        
        # Xử lý Spending và PPA cho Trial không phải Highlands
        if 'Trial' in df.columns and 'Spending' in df.columns:
            mask = df['Trial'] != 'Highlands Coffee'
            df.loc[mask, 'Spending'] = 0
            if 'PPA' in df.columns:
                df.loc[mask, 'PPA'] = -1
        
        # Tính lại PPA
        if 'PPA' in df.columns and 'Spending' in df.columns and 'Frequency_Visit' in df.columns:
            mask = (df['PPA'] != -1) & (df['Frequency_Visit'] > 0)
            df.loc[mask, 'PPA'] = df.loc[mask, 'Spending'] / df.loc[mask, 'Frequency_Visit']
        
        # Fill Segmentation dựa trên Spending
        if 'Spending' in df.columns:
            def get_segmentation(spending):
                if spending == -1:
                    return 'Seg.00 - Unknown'
                elif spending < 25000:
                    return 'Seg.01 - Mass (<VND 25K)'
                elif 25000 <= spending <= 59000:
                    return 'Seg.02 - Mass Asp (VND 25K - VND 59K)'
                elif 60000 <= spending <= 99000:
                    return 'Seg.03 - Premium (VND 60K - VND 99K)'
                else:
                    return 'Seg.04 - Super Premium (VND 100K+)'
            
            df['Segmentation'] = df['Spending'].apply(get_segmentation)
        
        # Fill NPS_P3M
        if 'NPS_P3M' not in df.columns:
            df['NPS_P3M'] = np.nan
        
        if 'Brand_Likability' in df.columns and 'Segmentation' in df.columns:
            mask_9 = (df['Brand_Likability'] == 'Highlands Coffee') | \
                     (df['Segmentation'].isin(['Seg.04 - Super Premium (VND 100K+)', 
                                               'Seg.03 - Premium (VND 60K - VND 99K)']))
            df.loc[mask_9 & df['NPS_P3M'].isna(), 'NPS_P3M'] = 9
            
            mask_8 = df['Segmentation'] == 'Seg.02 - Mass Asp (VND 25K - VND 59K)'
            df.loc[mask_8 & df['NPS_P3M'].isna(), 'NPS_P3M'] = 8
            
            # Fill phần còn lại bằng -1
            df['NPS_P3M'] = df['NPS_P3M'].fillna(-1)
        
        # Fill NPS_P3M_Group
        if 'NPS_P3M' in df.columns:
            def get_nps_group(score):
                if score >= 9:
                    return 'Promoter'
                elif score >= 8:
                    return 'Passive'
                else:
                    return 'Detractive'
            
            df['NPS_P3M_Group'] = df['NPS_P3M'].apply(get_nps_group)
        
        # Winsorize outliers
        if 'Frequency_Visit' in df.columns:
            df['Frequency_Visit'] = self._winsorize_series(df['Frequency_Visit'], 0.01, 0.99)
        
        if 'Spending' in df.columns:
            df['Spending'] = self._winsorize_series(df['Spending'], 0.01, 0.99)
        
        if 'NPS_P3M' in df.columns:
            df['NPS_P3M'] = self._winsorize_series(df['NPS_P3M'], 0.01, 0.99)
        
        logger.info(f"Đã làm sạch brandhealth: shape = {df.shape}")
        return df
    
    def clean_sa_var(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu SA variables.
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        logger.info("Đang làm sạch dữ liệu SA variables...")
        
        # Chuẩn hóa tên cột: Bỏ dấu # thay bằng _ và viết hoa chữ đầu
        rename_map = {}
        for col in df.columns:
            if '#' in col:
                # Thay # bằng _
                new_col = col.replace('#', '_')
                rename_map[col] = new_col
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Đã đổi tên cột: {rename_map}")
        
        # Drop rows với NaN trong Group_size
        if 'Group_size' in df.columns:
            df = df.dropna(subset=['Group_size'])
        
        # Drop Age_group nếu có
        if 'Age_group' in df.columns:
            df = df.drop(columns=['Age_group'])
        
        if 'Col' in df.columns:
            df = df.drop(columns=['Col'])

        # Drop các cột thu nhập không cần (GIỮ LẠI MPI_Mean và MPI_detail để tạo IncomeBand sau)
        cols_to_drop = ['MPI', 'MPI_2', 'MPI_Mean_Use']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Chuẩn hóa text
        text_cols = ['City', 'Occupation', 'Occupation_group', 'TOM', 'BUMO', 'MostFavourite']
        df = self._normalize_string_columns(df, text_cols)
        
        # Sửa lỗi chính tả brand
        for col in ['TOM', 'BUMO', 'MostFavourite']:
            if col in df.columns:
                df[col] = df[col].apply(self._fix_brand_spelling)
        
        # Xử lý thu nhập - Tạo IncomeBand từ MPI_detail và MPI_Mean
        if 'MPI_detail' in df.columns and 'MPI_Mean' in df.columns:
            df['IncomeBand'] = df['MPI_detail'].copy()
            mask = df['IncomeBand'].isna()
            df.loc[mask, 'IncomeBand'] = df.loc[mask, 'MPI_Mean'].apply(self._map_income_band_from_mean)
            df['IncomeBand'] = df['IncomeBand'].fillna('Unknown')
            df = df.drop(columns=['MPI_Mean', 'MPI_detail'])
        
        # Đổi tên Occupation_group thành Occupation_Group
        if 'Occupation_group' in df.columns:
            df = df.rename(columns={'Occupation_group': 'Occupation_Group'})
        
        # Đổi tên Age_Group_2 thành Age_Group_2 (đã được đổi từ Age#Group#2)
        if 'Age_Group_2' in df.columns:
            df = df.rename(columns={'Age_Group_2': 'Age_Group_2'})
        
        # Fill BUMO_Previous
        if 'BUMO_Previous' in df.columns:
            df['BUMO_Previous'] = df['BUMO_Previous'].fillna('Unknown')
        
        logger.info(f"Đã làm sạch SA variables: shape = {df.shape}")
        return df
    
    def clean_needstate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu needstate.
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()
        logger.info("Đang làm sạch dữ liệu needstate...")
        
        # Chuyển ID sang object
        df['ID'] = df['ID'].astype(str)
        
        # Kiểm tra độ dài ID và drop nếu bất thường
        mask = df['ID'].str.len() <= 20
        df = df[mask]
        
        # Loại bỏ duplicate rows
        df = df.drop_duplicates(keep='first')
        
        # Đổi tên cột
        if 'Day#Daypart' in df.columns:
            df = df.rename(columns={'Day#Daypart': 'Daypart'})
        
        # Chuẩn hóa text
        text_cols = [col for col in df.select_dtypes(include=['object']).columns]
        df = self._normalize_string_columns(df, text_cols)
        
        # Sửa chính tả Needstates
        if 'Needstates' in df.columns:
            df['Needstates'] = df['Needstates'].replace({
                'Socialzing': 'Socializing',
                'Enterntainment (watching movies. Playing games, browsing web,…)': 'Entertainment'
            })
        
        # Sửa NeedstateGroup
        if 'NeedstateGroup' in df.columns:
            df['NeedstateGroup'] = df['NeedstateGroup'].replace({
                'Relaxing & entertainment': 'Relaxing & Entertainment',
                'Working & business meeting': 'Working & Business meeting'
            })
        
        logger.info(f"Đã làm sạch needstate: shape = {df.shape}")
        return df
    
    def clean_all(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Làm sạch tất cả các datasets.
        
        Args:
            dfs: Dictionary các DataFrames với các key:
                 'customer_seg', 'brand_image', 'brandhealth', 'sa_var', 'needstate'
                 
        Returns:
            Dictionary các DataFrames đã làm sạch
        """
        logger.info("Đang bắt đầu làm sạch tất cả datasets...")
        
        cleaned = {}
        
        if 'customer_seg' in dfs:
            cleaned['customer_seg'] = self.clean_customer_segmentation(dfs['customer_seg'])
        
        if 'brand_image' in dfs:
            cleaned['brand_image'] = self.clean_brand_image(dfs['brand_image'])
        
        if 'brandhealth' in dfs:
            brand_image_df = cleaned.get('brand_image', dfs.get('brand_image'))
            cleaned['brandhealth'] = self.clean_brandhealth(dfs['brandhealth'], brand_image_df)
        
        if 'sa_var' in dfs:
            cleaned['sa_var'] = self.clean_sa_var(dfs['sa_var'])
        
        if 'needstate' in dfs:
            cleaned['needstate'] = self.clean_needstate(dfs['needstate'])
        
        logger.info("Đã làm sạch tất cả datasets thành công")
        return cleaned


def save_cleaned(dfs_clean: Dict[str, pd.DataFrame], base_dir: str = "data/processed/cleaned") -> None:
    """
    Lưu các DataFrames đã làm sạch ra file CSV.
    
    Args:
        dfs_clean: Dictionary các DataFrames đã làm sạch
        base_dir: Thư mục gốc để lưu các file đã làm sạch
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    filename_map = {
        'customer_seg': 'customer_segmentation_clean.csv',
        'brand_image': 'brand_image_clean.csv',
        'brandhealth': 'brandhealth_clean.csv',
        'sa_var': 'sa_var_clean.csv',
        'needstate': 'needstate_clean.csv'
    }
    
    for key, df in dfs_clean.items():
        if key in filename_map:
            filepath = base_path / filename_map[key]
            df.to_csv(filepath, index=False)
            logger.info(f"Đã lưu {key} vào {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    
    from src.preprocessing.data_loader import DataLoader
    
    # Load dữ liệu
    loader = DataLoader()
    raw_dfs = loader.load_all()
    
    # Ánh xạ sang các key mong muốn
    dfs = {
        'customer_seg': raw_dfs['segmentation'],
        'brand_image': raw_dfs['brand_image'],
        'brandhealth': raw_dfs['brandhealth'],
        'sa_var': raw_dfs['sa_var'],
        'needstate': raw_dfs['needstate']
    }
    
    # Làm sạch dữ liệu
    cleaner = DataCleaner()
    cleaned_dfs = cleaner.clean_all(dfs)
    
    # Lưu dữ liệu đã làm sạch
    save_cleaned(cleaned_dfs)
    
    print("\nHoàn tất quá trình làm sạch!")
    for key, df in cleaned_dfs.items():
        print(f"{key}: shape = {df.shape}")