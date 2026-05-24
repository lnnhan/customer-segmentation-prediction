"""
Feature Encoder Module for Highlands Coffee Customer Clustering Project

Module này thực hiện encoding cho dữ liệu sau feature engineering, 
chuẩn bị cho các thuật toán clustering.

Bao gồm:
- Log-transform cho biến numeric skewed (Visit, Spending, PPA)
- Standard scaling cho biến numeric thông thường
- Ordinal encoding cho biến có thứ tự (Segmentation, Age_Group_2, Comprehension)
- One-hot encoding cho biến categorical (City, Gender, Occupation_Group, etc.)

Author: Senior ML Engineer
Date: December 2025
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OrdinalEncoder, 
    OneHotEncoder, 
    StandardScaler, 
    MinMaxScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Import encoding configurations from config
from src.config import (
    DROP_COLS,
    LOG_NUMERIC_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    CATEGORICAL_COLS,
    ORDINAL_CATEGORIES
)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class EncoderConfig:
    """
    Cấu hình cho FeatureEncoder
    
    Attributes:
        scaler_type: Loại scaler cho numeric ('standard' hoặc 'minmax')
        handle_unknown: Cách xử lý giá trị unknown trong categorical
        sparse_output: Output có phải sparse matrix không
    """
    scaler_type: str = 'standard'  # 'standard' hoặc 'minmax'
    handle_unknown: str = 'ignore'
    sparse_output: bool = False


# ============================================================================
# CUSTOM TRANSFORMER: LOG TRANSFORM
# ============================================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer để apply log1p transformation
    
    Dùng cho các biến có phân phối skewed như Visit, Spending, PPA
    
    QUAN TRỌNG: Xử lý giá trị -1 (missing) trước khi log transform
    - Giá trị -1 được thay bằng 0 trước khi log
    - log1p(0) = 0 (an toàn)
    """
    
    def fit(self, X, y=None):
        """Fit method (không làm gì vì log transform không cần fit)"""
        return self
    
    def transform(self, X):
        """
        Apply log1p transformation
        
        Xử lý:
        1. Thay giá trị âm (thường là -1 cho missing) thành 0
        2. Apply log1p để tránh log(0)
        """
        X_copy = np.array(X, dtype=np.float64, copy=True)
        
        # Thay tất cả giá trị âm thành 0 (an toàn cho log1p)
        X_copy[X_copy < 0] = 0
        
        # Apply log1p
        return np.log1p(X_copy)
    
    def inverse_transform(self, X):
        """Inverse log1p transformation"""
        return np.expm1(X)


# ============================================================================
# MAIN FEATURE ENCODER CLASS
# ============================================================================

class FeatureEncoder:
    """
    Feature Encoder cho Highlands Coffee Customer Clustering Project
    
    Class này thực hiện encoding cho DataFrame sau feature engineering,
    chuẩn bị dữ liệu cho các thuật toán clustering.
    
    Phân loại cột:
    - DROP: ID, Year, TOM, BUMO, Brand, Spontaneous, Trial, P3M, P1M, etc.
    - LOG_NUMERIC: Visit, Spending, PPA (log1p + scale)
    - NUMERIC: Age, Group_size, NPS_P3M, Brand_Loyalty, các flag, etc. (scale only)
    - ORDINAL: Segmentation, Age_Group_2, Comprehension (ordinal encoding)
    - CATEGORICAL: City, Gender, Occupation_Group, Daypart, etc. (one-hot encoding)
    
    Attributes:
        config: EncoderConfig object chứa cấu hình
        preprocessor_: ColumnTransformer đã fit
        feature_names_: List tên feature sau khi encode
    
    Example:
        >>> encoder = FeatureEncoder()
        >>> X = encoder.fit_transform(df_final)
        >>> print(encoder.feature_names_[:10])
    """
    
    # ========================================================================
    # NOTE: Column definitions moved to src/config.py for better reusability
    # ========================================================================
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        """
        Khởi tạo FeatureEncoder
        
        Args:
            config: EncoderConfig object. Nếu None, dùng config mặc định
        """
        self.config = config if config is not None else EncoderConfig()
        self.preprocessor_ = None
        self.feature_names_ = None
    
    def _build_preprocessor(self):
        """
        Xây dựng ColumnTransformer để xử lý tất cả các nhóm cột
        
        Returns:
            ColumnTransformer đã cấu hình
        """
        # Chọn scaler type
        if self.config.scaler_type == 'minmax':
            scaler_class = MinMaxScaler
        else:
            scaler_class = StandardScaler
        
        # 1. Pipeline cho LOG_NUMERIC: log1p + scale
        log_num_pipeline = Pipeline([
            ('log', LogTransformer()),
            ('scale', scaler_class())
        ])
        
        # 2. Scaler cho NUMERIC thông thường
        num_pipeline = Pipeline([
            ('scale', scaler_class())
        ])
        
        # 3. OrdinalEncoder cho ORDINAL
        ordinal_encoder = OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        # 4. OneHotEncoder cho CATEGORICAL
        onehot_encoder = OneHotEncoder(
            handle_unknown=self.config.handle_unknown,
            sparse_output=self.config.sparse_output,
            drop='first'  # Drop first category để tránh multicollinearity
        )
        
        # 5. Tạo ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('log_num', log_num_pipeline, LOG_NUMERIC_COLS),
                ('num', num_pipeline, NUMERIC_COLS),
                ('ord', ordinal_encoder, ORDINAL_COLS),
                ('cat', onehot_encoder, CATEGORICAL_COLS)
            ],
            remainder='drop',  # Drop các cột không được chỉ định
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị DataFrame: drop các cột không cần thiết
        
        Args:
            df: DataFrame gốc
            
        Returns:
            DataFrame đã drop các cột trong DROP_COLS
        """
        df_prep = df.copy()
        
        # Drop các cột không dùng (nếu có trong df)
        cols_to_drop = [col for col in DROP_COLS if col in df_prep.columns]
        if cols_to_drop:
            df_prep = df_prep.drop(columns=cols_to_drop)
            print(f"Đã drop {len(cols_to_drop)} cột: {cols_to_drop[:5]}...")
        
        return df_prep
    
    def _extract_feature_names(self):
        """
        Trích xuất tên features sau khi transform từ preprocessor
        
        Returns:
            List các tên feature
        """
        feature_names = []
        
        # Lấy feature names từ ColumnTransformer
        for name, transformer, columns in self.preprocessor_.transformers_:
            if name == 'remainder':
                continue
            
            if name == 'log_num' or name == 'num':
                # Numeric features giữ nguyên tên cột
                feature_names.extend(columns)
            
            elif name == 'ord':
                # Ordinal features giữ nguyên tên cột
                feature_names.extend(columns)
            
            elif name == 'cat':
                # OneHot features: lấy từ encoder
                try:
                    cat_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(cat_features)
                except AttributeError:
                    # Fallback nếu không có get_feature_names_out
                    feature_names.extend(columns)
        
        return feature_names
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit encoder trên data và transform
        
        Args:
            df: DataFrame sau feature engineering (df_final)
            
        Returns:
            X: numpy array đã được encode, shape (n_samples, n_features)
        """
        print("=" * 70)
        print("FEATURE ENCODING - FIT & TRANSFORM")
        print("=" * 70)
        print(f"\nInput shape: {df.shape}")
        
        # 1. Chuẩn bị DataFrame (drop các cột không dùng)
        print("\n[1/3] Chuẩn bị dữ liệu...")
        df_prep = self._prepare_dataframe(df)
        print(f"Shape sau khi drop: {df_prep.shape}")
        
        # 2. Build preprocessor
        print("\n[2/3] Xây dựng preprocessing pipeline...")
        self.preprocessor_ = self._build_preprocessor()
        
        # 3. Fit & Transform
        print("\n[3/3] Fit và transform dữ liệu...")
        print(f"  - Log + Scale: {len(LOG_NUMERIC_COLS)} cột")
        print(f"  - Scale only: {len(NUMERIC_COLS)} cột")
        print(f"  - Ordinal encode: {len(ORDINAL_COLS)} cột")
        print(f"  - One-hot encode: {len(CATEGORICAL_COLS)} cột")
        
        X = self.preprocessor_.fit_transform(df_prep)
        
        # 4. Lưu feature names
        self.feature_names_ = self._extract_feature_names()
        
        print("\n" + "=" * 70)
        print(f"✓ ENCODING HOÀN TẤT!")
        print(f"Output shape: {X.shape}")
        print(f"Số features: {len(self.feature_names_)}")
        print("=" * 70)
        
        return X
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dữ liệu mới với encoder đã fit
        
        Args:
            df: DataFrame cần encode
            
        Returns:
            X: numpy array đã được encode
        """
        if self.preprocessor_ is None:
            raise ValueError(
                "Encoder chưa được fit! Hãy gọi fit_transform() trước khi gọi transform()."
            )
        
        print("Transforming new data...")
        
        # Chuẩn bị DataFrame
        df_prep = self._prepare_dataframe(df)
        
        # Transform
        X = self.preprocessor_.transform(df_prep)
        
        print(f"✓ Transform hoàn tất. Output shape: {X.shape}")
        
        return X
    
    def save(self, filepath: str = '../../models_saved/encoders/feature_encoder.pkl'):
        """
        Lưu toàn bộ FeatureEncoder (bao gồm preprocessor và feature_names)
        
        Args:
            filepath: Đường dẫn file để lưu encoder
        """
        if self.preprocessor_ is None:
            raise ValueError("Encoder chưa được fit! Không thể lưu.")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Lưu toàn bộ object
        joblib.dump(self, filepath)
        
        print(f"✓ Đã lưu FeatureEncoder vào: {filepath}")
    
    @staticmethod
    def load(filepath: str = '../../models_saved/encoders/feature_encoder.pkl'):
        """
        Load FeatureEncoder từ file
        
        Args:
            filepath: Đường dẫn file chứa encoder
            
        Returns:
            FeatureEncoder đã được load
        """
        encoder = joblib.load(filepath)
        print(f"✓ Đã load FeatureEncoder từ: {filepath}")
        return encoder
    
    def get_feature_names(self) -> List[str]:
        """
        Trả về danh sách tên features sau encoding
        
        Returns:
            List tên features
        """
        if self.feature_names_ is None:
            raise ValueError("Feature names chưa được tạo. Hãy gọi fit_transform() trước.")
        
        return self.feature_names_.copy()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HIGHLANDS COFFEE - FEATURE ENCODING DEMO")
    print("="*70 + "\n")
    
    # 1. Đọc dữ liệu
    print("[1] Đọc dữ liệu từ feature_engineering_data.csv...")
    df_final = pd.read_csv('../../data/processed/feature_engineering_data.csv')
    print(f"✓ Đã load {df_final.shape[0]} rows, {df_final.shape[1]} columns\n")
    
    # 2. Tạo encoder với config mặc định
    print("[2] Khởi tạo FeatureEncoder...")
    config = EncoderConfig(
        scaler_type='standard',  # Có thể đổi thành 'minmax'
        handle_unknown='ignore',
        sparse_output=False
    )
    encoder = FeatureEncoder(config=config)
    print("✓ Encoder đã sẵn sàng\n")
    
    # 3. Fit và transform
    print("[3] Fit & Transform dữ liệu...")
    try:
        X = encoder.fit_transform(df_final)
        
        print(f"\n[4] Thông tin về ma trận encoded:")
        print(f"  - Shape: {X.shape}")
        print(f"  - Type: {type(X)}")
        print(f"  - Số features: {len(encoder.feature_names_)}")
        
        # Hiển thị một số feature names
        print(f"\n[5] Top 20 feature names:")
        for i, name in enumerate(encoder.feature_names_[:20], 1):
            print(f"  {i:2d}. {name}")
        
        if len(encoder.feature_names_) > 20:
            print(f"  ... và {len(encoder.feature_names_) - 20} features khác")
        
        # 4. Lưu encoder
        print(f"\n[6] Lưu encoder...")
        encoder.save()
        
        # 5. Lưu ma trận X ra CSV (optional - để kiểm tra)
        print(f"\n[7] Lưu ma trận encoded ra file...")
        output_path = '../../data/processed/encoded_data.csv'
        df_encoded = pd.DataFrame(X, columns=encoder.feature_names_)
        df_encoded.to_csv(output_path, index=False)
        print(f"✓ Đã lưu vào: {output_path}")
        
        # 6. Test load encoder
        print(f"\n[8] Test load encoder...")
        encoder_loaded = FeatureEncoder.load()
        print(f"✓ Load thành công! Feature names count: {len(encoder_loaded.feature_names_)}")
        
        print("\n" + "="*70)
        print("✓ DEMO HOÀN TẤT!")
        print("="*70 + "\n")
        
        print("Bạn có thể sử dụng encoder như sau:")
        print("  >>> from encoder import FeatureEncoder")
        print("  >>> encoder = FeatureEncoder.load()")
        print("  >>> X = encoder.transform(df_new)")
        print("  >>> feature_names = encoder.get_feature_names()")
        
    except Exception as e:
        print(f"\n✗ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()
