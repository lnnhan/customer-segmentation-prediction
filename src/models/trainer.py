"""
Model Trainer Module for Coffee Project

Module này huấn luyện các mô hình clustering (KMeans, GMM, DBSCAN, HDBSCAN) cho bài toán
phân cụm khách hàng Highlands Coffee.

Tính năng chính:
- Load dữ liệu đã encode
- Train model với cấu hình cố định
- Đánh giá bằng 3 metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
- Lưu/load model và labels
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
import joblib

from src.models.evaluator import ClusteringEvaluator


@dataclass
class TrainingConfig:
    """
    Cấu hình cho quá trình training clustering models
    
    Attributes:
        data_path: Đường dẫn file CSV đã encode
        model_type: Loại model ('kmeans', 'gmm', 'dbscan', 'hdbscan')
        n_clusters: Số cụm (dùng cho KMeans/GMM)
        random_state: Random seed để reproducible
        selected_features: Danh sách tên cột cần chọn. Nếu None, dùng tất cả
        results_path: Đường dẫn lưu kết quả đánh giá (CSV)
        model_path: Đường dẫn lưu model (PKL)
        model_params: Dict chứa hyperparams riêng cho từng model
            - KMeans: {"n_init": 20, "max_iter": 500}
            - GMM: {"covariance_type": "full"}
            - DBSCAN: {"eps": 0.7, "min_samples": 10}
            - HDBSCAN: {"min_cluster_size": 15, "min_samples": 10}
    """
    data_path: str = "data/processed/encoded_data.csv"
    model_type: str = "kmeans"  # 'kmeans', 'gmm', 'dbscan', 'hdbscan'
    n_clusters: int = 5
    random_state: int = 42
    selected_features: Optional[List[str]] = None
    results_path: str = "results/cluster_results.csv"
    model_path: str = "results/best_cluster_model.pkl"
    model_params: Dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """
    Class để huấn luyện các mô hình clustering
    
    Workflow:
    1. load_data()      - Load dữ liệu đã encode
    2. build_model()    - Khởi tạo model theo config
    3. train_model()    - Fit model với cấu hình cố định
    4. evaluate()       - Đánh giá model hiện tại
    5. save_model()     - Lưu model
    
    Attributes:
        config: TrainingConfig object
        evaluator: ClusteringEvaluator object
        logger: Logger instance
        df: DataFrame chứa dữ liệu
        X: Numpy array features
        model: Model đã train
    
    Example:
        >>> config = TrainingConfig(
        ...     model_type='kmeans',
        ...     n_clusters=5,
        ...     model_params={"n_init": 20}
        ... )
        >>> evaluator = ClusteringEvaluator()
        >>> trainer = ModelTrainer(config, evaluator)
        >>> trainer.load_data()
        >>> trainer.train_model()
        >>> metrics = trainer.evaluate()
        >>> trainer.save_model()
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        evaluator: ClusteringEvaluator,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo ModelTrainer
        
        Args:
            config: TrainingConfig object chứa cấu hình
            evaluator: ClusteringEvaluator để đánh giá models
            logger: Logger instance. Nếu None, tạo logger mặc định
        """
        self.config = config
        self.evaluator = evaluator
        
        # Khởi tạo logger
        if logger is None:
            self.logger = logging.getLogger("ModelTrainer")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        else:
            self.logger = logger
        
        # Khởi tạo attributes
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.model: Optional[Union[KMeans, GaussianMixture, DBSCAN, 'HDBSCAN']] = None
        
        self.logger.info("✓ ModelTrainer initialized")
        self.logger.info(f"  Model type: {self.config.model_type}")
        self.logger.info(f"  Data path: {self.config.data_path}")
    
    def load_data(self) -> None:
        """
        Load dữ liệu đã encode từ CSV
        
        Đọc file CSV, chọn features nếu cần, chuyển thành numpy array
        """
        self.logger.info(f"📂 Loading data from {self.config.data_path}...")
        
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"File không tồn tại: {self.config.data_path}")
        
        # Load CSV
        self.df = pd.read_csv(self.config.data_path)
        
        # Chọn lọc features nếu cần
        if self.config.selected_features is not None:
            missing_cols = set(self.config.selected_features) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Các cột không tồn tại: {missing_cols}")
            
            self.df = self.df[self.config.selected_features]
            self.logger.info(f"  ✓ Selected {len(self.config.selected_features)} features")
        
        # Chuyển thành numpy array
        self.X = self.df.values
        
        self.logger.info(f"  ✓ Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
    
    def build_model(self, n_clusters: Optional[int] = None) -> Union[KMeans, GaussianMixture, DBSCAN, 'HDBSCAN']:
        """
        Khởi tạo mô hình clustering với cấu hình từ config
        
        Args:
            n_clusters: Số cụm (dùng cho KMeans/GMM). Nếu None, dùng config.n_clusters
        
        Returns:
            Model instance đã khởi tạo
        
        Raises:
            ValueError: Nếu model_type không hợp lệ
            ImportError: Nếu HDBSCAN chưa được cài đặt
        """
        model_type = self.config.model_type.lower()
        params = dict(self.config.model_params)  # Copy để không sửa dict gốc
        
        if model_type == "kmeans":
            default_params = {
                "n_clusters": n_clusters or self.config.n_clusters,
                "n_init": "auto",
                "random_state": self.config.random_state,
            }
            default_params.update(params)
            return KMeans(**default_params)
        
        elif model_type == "gmm":
            default_params = {
                "n_components": n_clusters or self.config.n_clusters,
                "random_state": self.config.random_state,
            }
            default_params.update(params)
            return GaussianMixture(**default_params)
        
        elif model_type == "dbscan":
            default_params = {
                "eps": 0.5,
                "min_samples": 5,
                "n_jobs": -1,
            }
            default_params.update(params)
            return DBSCAN(**default_params)
        
        elif model_type == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise ImportError(
                    "HDBSCAN chưa được cài đặt. Cài bằng: pip install hdbscan"
                )
            default_params = {
                "min_cluster_size": 5,
                "min_samples": None,
                "core_dist_n_jobs": -1,
            }
            default_params.update(params)
            return HDBSCAN(**default_params)
        
        else:
            raise ValueError(
                f"model_type '{self.config.model_type}' không hợp lệ. "
                f"Chọn 'kmeans', 'gmm', 'dbscan', hoặc 'hdbscan'."
            )
    
    def train_model(self) -> None:
        """
        Huấn luyện mô hình với cấu hình đã cho
        
        Build model từ config và fit trên toàn bộ dữ liệu X
        """
        if self.X is None:
            raise ValueError("Chưa load dữ liệu! Gọi load_data() trước.")
        
        model_type = self.config.model_type.lower()
        
        if model_type in ['dbscan', 'hdbscan']:
            self.logger.info(f"🔧 Training {model_type.upper()}...")
        else:
            self.logger.info(f"🔧 Training {model_type.upper()} with {self.config.n_clusters} clusters...")
        
        self.model = self.build_model()
        self.model.fit(self.X)
        
        self.logger.info("  ✓ Model trained successfully")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Đánh giá mô hình hiện tại bằng ClusteringEvaluator
        
        Returns:
            Dict chứa các metrics:
            {
                'model': str,
                'n_clusters': int,
                'silhouette': float,
                'calinski_harabasz': float,
                'davies_bouldin': float
            }
        """
        if self.model is None:
            raise ValueError("Chưa có model! Gọi train_model() trước.")
        
        if self.X is None:
            raise ValueError("Chưa load dữ liệu! Gọi load_data() trước.")
        
        self.logger.info("📊 Evaluating model...")
        
        # Lấy labels
        labels = self.get_cluster_labels()
        
        # Đánh giá
        metrics = self.evaluator.evaluate_once(
            X=self.X,
            labels=labels,
            model_name=self.config.model_type
        )
        
        # Log metrics
        self.logger.info(f"  ✓ Silhouette Score    : {metrics['silhouette']:>7.4f}")
        self.logger.info(f"  ✓ Calinski-Harabasz   : {metrics['calinski_harabasz']:>7.2f}")
        self.logger.info(f"  ✓ Davies-Bouldin Index: {metrics['davies_bouldin']:>7.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Lưu model vào file PKL
        
        Args:
            path: Đường dẫn file PKL. Nếu None, dùng config.model_path
        """
        if self.model is None:
            raise ValueError("Chưa có model để lưu! Gọi train_model() trước.")
        
        save_path = path if path is not None else self.config.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        joblib.dump(self.model, save_path)
        self.logger.info(f"💾 Model saved: {save_path}")
    
    @staticmethod
    def load_model(path: str) -> Union[KMeans, GaussianMixture, DBSCAN, 'HDBSCAN']:
        """
        Load model từ file PKL
        
        Args:
            path: Đường dẫn file PKL
        
        Returns:
            Model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File không tồn tại: {path}")
        
        return joblib.load(path)
    
    def get_cluster_labels(self) -> np.ndarray:
        """
        Lấy cluster labels từ model
        
        Returns:
            Numpy array chứa cluster labels
            (DBSCAN/HDBSCAN có thể có label -1 cho noise points)
        """
        if self.model is None:
            raise ValueError("Chưa có model! Gọi train_model() trước.")
        
        if self.X is None:
            raise ValueError("Chưa load dữ liệu! Gọi load_data() trước.")
        
        if self.config.model_type.lower() in ['dbscan', 'hdbscan']:
            return self.model.fit_predict(self.X)
        elif hasattr(self.model, 'labels_'):
            return self.model.labels_
        else:
            return self.model.predict(self.X)
    
    def save_labels(self, output_path: str) -> None:
        """
        Lưu cluster labels ra file CSV
        
        Args:
            output_path: Đường dẫn file CSV để lưu labels
        """
        labels = self.get_cluster_labels()
        
        df_labels = self.df.copy()
        df_labels['cluster'] = labels
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_labels.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"💾 Labels saved: {output_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demo script để test ModelTrainer
    """
    from src.models.evaluator import ClusteringEvaluator
    
    # Example 1: Train KMeans với custom params
    config_kmeans = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="kmeans",
        n_clusters=5,
        model_params={"n_init": 20, "max_iter": 500}
    )
    
    evaluator = ClusteringEvaluator()
    trainer = ModelTrainer(config=config_kmeans, evaluator=evaluator)
    
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    trainer.save_model()
    trainer.save_labels("results/kmeans_labels.csv")
    
    print("\n" + "="*70)
    print("KMeans Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Example 2: Train DBSCAN
    config_dbscan = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="dbscan",
        model_params={"eps": 2.0, "min_samples": 10}
    )
    
    trainer_dbscan = ModelTrainer(config=config_dbscan, evaluator=evaluator)
    trainer_dbscan.load_data()
    trainer_dbscan.train_model()
    metrics_dbscan = trainer_dbscan.evaluate()
    
    print("\n" + "="*70)
    print("DBSCAN Metrics:")
    for key, value in metrics_dbscan.items():
        print(f"  {key}: {value}")
    
    # Example 3: Train GMM
    config_gmm = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="gmm",
        n_clusters=4,
        model_params={"covariance_type": "full"}
    )
    
    trainer_gmm = ModelTrainer(config=config_gmm, evaluator=evaluator)
    trainer_gmm.load_data()
    trainer_gmm.train_model()
    metrics_gmm = trainer_gmm.evaluate()
    
    print("\n" + "="*70)
    print("GMM Metrics:")
    for key, value in metrics_gmm.items():
        print(f"  {key}: {value}")

