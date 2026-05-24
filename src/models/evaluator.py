"""
Clustering Evaluator Module for Coffee Project

Module này đánh giá chất lượng các mô hình clustering (phân cụm khách hàng)
bằng 3 chỉ số chính:
- Silhouette Score (cao hơn tốt hơn)
- Calinski-Harabasz Score (cao hơn tốt hơn)
- Davies-Bouldin Index (thấp hơn tốt hơn)
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# ============================================================================
# ABSTRACT BASE CLASS - TRỪU TƯỢNG
# ============================================================================

class BaseMetric(ABC):
    """
    Abstract Base Class cho tất cả clustering metrics
    
    Thể hiện tính TRỪU TƯỢNG trong OOP:
    - Không thể khởi tạo trực tiếp
    - Định nghĩa interface chung cho tất cả metrics
    - Các class con phải implement abstract methods
    """
    
    def __init__(self, name: str, higher_is_better: bool = True):
        """
        Khởi tạo base metric
        
        Args:
            name: Tên metric
            higher_is_better: True nếu giá trị cao hơn tốt hơn, False nếu ngược lại
        """
        self.name = name
        self.higher_is_better = higher_is_better
    
    @abstractmethod
    def compute(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Tính toán metric (ABSTRACT METHOD - bắt buộc override)
        
        Args:
            X: Feature matrix
            labels: Cluster labels
        
        Returns:
            Giá trị metric
        """
        pass
    
    def is_better(self, score1: float, score2: float) -> bool:
        """
        So sánh 2 scores xem score1 có tốt hơn score2 không
        
        Args:
            score1: Score thứ nhất
            score2: Score thứ hai
        
        Returns:
            True nếu score1 tốt hơn score2
        """
        if self.higher_is_better:
            return score1 > score2
        else:
            return score1 < score2
    
    def __repr__(self) -> str:
        direction = "↑" if self.higher_is_better else "↓"
        return f"{self.name} ({direction} better)"


# ============================================================================
# CONCRETE METRIC CLASSES - KẾ THỪA & ĐA HÌNH
# ============================================================================

class SilhouetteMetric(BaseMetric):
    """
    Silhouette Score Metric
    
    Kế thừa từ BaseMetric và override compute() - thể hiện ĐA HÌNH
    """
    
    def __init__(self):
        super().__init__(name="Silhouette Score", higher_is_better=True)
    
    def compute(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Override compute() - ĐA HÌNH"""
        return float(silhouette_score(X, labels))


class CalinskiMetric(BaseMetric):
    """
    Calinski-Harabasz Score Metric
    
    Kế thừa từ BaseMetric và override compute() - thể hiện ĐA HÌNH
    """
    
    def __init__(self):
        super().__init__(name="Calinski-Harabasz Score", higher_is_better=True)
    
    def compute(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Override compute() - ĐA HÌNH"""
        return float(calinski_harabasz_score(X, labels))


class DaviesMetric(BaseMetric):
    """
    Davies-Bouldin Index Metric
    
    Kế thừa từ BaseMetric và override compute() - thể hiện ĐA HÌNH
    """
    
    def __init__(self):
        super().__init__(name="Davies-Bouldin Index", higher_is_better=False)
    
    def compute(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Override compute() - ĐA HÌNH"""
        return float(davies_bouldin_score(X, labels))


# ============================================================================
# MAIN EVALUATOR CLASS
# ============================================================================

class ClusteringEvaluator:
    """
    Class để đánh giá chất lượng phân cụm (clustering evaluation)
    
    Tính toán 3 chỉ số chính:
    - Silhouette Score: đo độ cohesion (cụm chặt) và separation (tách biệt)
    - Calinski-Harabasz Score: tỉ lệ between-cluster/within-cluster variance
    - Davies-Bouldin Index: mức độ overlap giữa các cụm
    
    Attributes:
        logger: Logger instance để ghi log
        results_: List các dict chứa kết quả đánh giá từng model
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Khởi tạo ClusteringEvaluator
        
        Args:
            logger: Logger instance. Nếu None, tạo logger mặc định với level INFO
        """
        if logger is None:
            self.logger = logging.getLogger("ClusteringEvaluator")
            self.logger.setLevel(logging.INFO)
            
            # Tạo console handler nếu chưa có
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                # Chỉ hiển thị message
                formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            self.logger.propagate = False  # Tránh log trùng lặp
        else:
            self.logger = logger
        
        # Khởi tạo các metric objects (sử dụng KẾ THỪA & ĐA HÌNH)
        self.metrics: Dict[str, BaseMetric] = {
            'silhouette': SilhouetteMetric(),
            'calinski_harabasz': CalinskiMetric(),
            'davies_bouldin': DaviesMetric()
        }
        
        # Lưu kết quả đánh giá
        self.results_: List[Dict[str, Any]] = []
    
    def evaluate_once(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        labels: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Đánh giá một mô hình clustering bằng 3 metric chính
        
        Args:
            X: Feature matrix đã encode, shape (n_samples, n_features)
            labels: Nhãn cụm cho mỗi sample, shape (n_samples,)
            model_name: Tên mô hình (e.g., "KMeans_k5", "GMM_k4")
        
        Returns:
            Dict chứa các metric:
            {
                'model': str,
                'n_clusters': int,
                'silhouette': float,
                'calinski_harabasz': float,
                'davies_bouldin': float
            }
        
        Raises:
            ValueError: Nếu số cụm < 2 hoặc tất cả labels giống nhau
        """
        # Không log nữa, để trainer.py quản lý log
        # self.logger.info(f"Đánh giá mô hình: {model_name}")
        
        # Chuyển DataFrame thành numpy array nếu cần
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        labels_array = np.array(labels)
        
        # Kiểm tra tính hợp lệ của labels
        n_unique_labels = len(np.unique(labels_array))
        
        if n_unique_labels < 2:
            error_msg = f"Số cụm phải >= 2, nhưng chỉ có {n_unique_labels} cụm duy nhất"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(labels_array) != X_array.shape[0]:
            error_msg = f"Số lượng labels ({len(labels_array)}) không khớp với số samples ({X_array.shape[0]})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Tính các metric (sử dụng ĐA HÌNH - mỗi metric compute() khác nhau)
        try:
            result = {
                'model': model_name,
                'n_clusters': n_unique_labels,
            }
            
            # Dùng polymorphism: gọi compute() của từng metric object
            for metric_name, metric_obj in self.metrics.items():
                result[metric_name] = metric_obj.compute(X_array, labels_array)
            
            # Thêm vào results_
            self.results_.append(result)
            
           
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính metric cho {model_name}: {str(e)}")
            raise
    
    def evaluate_many(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        labels_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Đánh giá nhiều mô hình clustering cùng lúc
        
        Args:
            X: Feature matrix đã encode
            labels_dict: Dict mapping {model_name: labels}
                Example: {
                    "KMeans_k5": labels_kmeans,
                    "GMM_k5": labels_gmm,
                    "Hierarchical_k5": labels_hc
                }
        
        Returns:
            DataFrame tổng hợp kết quả của tất cả models
            Columns: ['model', 'n_clusters', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
        """
        self.logger.info(f"Đánh giá {len(labels_dict)} mô hình clustering...")
        
        for model_name, labels in labels_dict.items():
            try:
                self.evaluate_once(X, labels, model_name=model_name)
            except Exception as e:
                self.logger.warning(f"Bỏ qua {model_name} do lỗi: {str(e)}")
                continue
        
        # Tạo DataFrame từ results_
        if not self.results_:
            self.logger.warning("Không có kết quả nào để tạo DataFrame")
            return pd.DataFrame()
        
        df_results = pd.DataFrame(self.results_)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"\n{df_results.to_string(index=False)}\n")
        
        return df_results
    
    def save_results(self, filepath: str) -> None:
        """
        Lưu kết quả đánh giá ra file CSV
        
        Args:
            filepath: Đường dẫn file CSV để lưu (e.g., "models_saved/cluster_eval.csv")
        """
        if not self.results_:
            self.logger.warning("Chưa có kết quả đánh giá nào để lưu. Gọi evaluate_once() hoặc evaluate_many() trước.")
            return
        
        df_results = pd.DataFrame(self.results_)
        
        # Tạo thư mục nếu chưa tồn tại
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"✓ Đã lưu kết quả vào: {filepath}")
    
    def plot_scores(
        self,
        metric: str = "silhouette",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Vẽ bar chart so sánh metric giữa các mô hình
        
        Args:
            metric: Tên metric cần vẽ. Các giá trị hợp lệ:
                - 'silhouette': Silhouette Score (cao hơn tốt hơn)
                - 'calinski_harabasz': Calinski-Harabasz Score (cao hơn tốt hơn)
                - 'davies_bouldin': Davies-Bouldin Index (thấp hơn tốt hơn)
            figsize: Kích thước figure (width, height)
            save_path: Đường dẫn để lưu hình. Nếu None, chỉ hiển thị không lưu
        
        Raises:
            ValueError: Nếu metric không hợp lệ
        """
        # Validate metric
        valid_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        if metric not in valid_metrics:
            raise ValueError(
                f"Metric '{metric}' không hợp lệ. Chọn một trong: {valid_metrics}"
            )
        
        if not self.results_:
            self.logger.warning("Chưa có kết quả để vẽ biểu đồ. Gọi evaluate_once() hoặc evaluate_many() trước.")
            return
        
        # Tạo DataFrame
        df = pd.DataFrame(self.results_)
        
        # Kiểm tra metric có trong DataFrame không
        if metric not in df.columns:
            self.logger.error(f"Metric '{metric}' không tồn tại trong kết quả")
            return
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Vẽ bar chart
        models = df['model'].values
        scores = df[metric].values
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.2)
        
        # Thêm giá trị trên mỗi bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Thiết lập labels và title
        metric_labels = {
            'silhouette': 'Silhouette Score (↑ better)',
            'calinski_harabasz': 'Calinski-Harabasz Score (↑ better)',
            'davies_bouldin': 'Davies-Bouldin Index (↓ better)'
        }
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_title(
            f'Clustering Evaluation: {metric_labels[metric]}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Xoay labels trục x nếu tên model dài
        plt.xticks(rotation=45, ha='right')
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Lưu hoặc hiển thị
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Đã lưu biểu đồ vào: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_scores(
        self,
        figsize: tuple = (15, 5),
        save_path: Optional[str] = None
    ) -> None:
        """
        Vẽ tất cả 3 metric trong một figure (3 subplots)
        
        Args:
            figsize: Kích thước figure (width, height)
            save_path: Đường dẫn để lưu hình. Nếu None, chỉ hiển thị
        """
        if not self.results_:
            self.logger.warning("Chưa có kết quả để vẽ biểu đồ.")
            return
        
        df = pd.DataFrame(self.results_)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        titles = [
            'Silhouette Score (↑ better)',
            'Calinski-Harabasz Score (↑ better)',
            'Davies-Bouldin Index (↓ better)'
        ]
        
        for ax, metric, title in zip(axes, metrics, titles):
            if metric not in df.columns:
                continue
            
            models = df['model'].values
            scores = df[metric].values
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
            bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.2)
            
            # Thêm giá trị
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Model', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Đã lưu biểu đồ tổng hợp vào: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_best_model(self, metric: str = "silhouette") -> Dict[str, Any]:
        """
        Tìm mô hình tốt nhất dựa trên một metric
        
        Args:
            metric: Metric để so sánh ('silhouette', 'calinski_harabasz', 'davies_bouldin')
        
        Returns:
            Dict chứa thông tin mô hình tốt nhất
        """
        if not self.results_:
            self.logger.warning("Chưa có kết quả để tìm mô hình tốt nhất")
            return {}
        
        df = pd.DataFrame(self.results_)
        
        # Davies-Bouldin: thấp hơn tốt hơn, còn lại cao hơn tốt hơn
        if metric == 'davies_bouldin':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        
        best_model = df.loc[best_idx].to_dict()
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"MÔ HÌNH TỐT NHẤT (theo {metric}):")
        self.logger.info(f"{'='*70}")
        for key, value in best_model.items():
            self.logger.info(f"  {key}: {value}")
        
        return best_model
    
    def clear_results(self) -> None:
        """Xóa tất cả kết quả đã lưu"""
        self.results_ = []
        self.logger.info("Đã xóa tất cả kết quả đánh giá")


"""
Example Usage:

from src.models.evaluator import ClusteringEvaluator
import numpy as np

# 1. Khởi tạo evaluator
evaluator = ClusteringEvaluator()

# 2. Đánh giá một mô hình
X = np.random.rand(1000, 20)  # 1000 samples, 20 features
labels_kmeans = np.random.randint(0, 5, 1000)  # 5 clusters

results = evaluator.evaluate_once(X, labels_kmeans, model_name="KMeans_k5")
print(results)
# Output: {'model': 'KMeans_k5', 'n_clusters': 5, 'silhouette': 0.45, ...}

# 3. Đánh giá nhiều mô hình
labels_gmm = np.random.randint(0, 5, 1000)
labels_hc = np.random.randint(0, 4, 1000)

results_df = evaluator.evaluate_many(X, {
    "KMeans_k5": labels_kmeans,
    "GMM_k5": labels_gmm,
    "Hierarchical_k4": labels_hc
})
print(results_df)

# 4. Lưu kết quả
evaluator.save_results("models_saved/cluster_eval.csv")

# 5. Vẽ biểu đồ so sánh
evaluator.plot_scores(
    metric="silhouette",
    save_path="reports/figures/silhouette_compare.png"
)

# 6. Vẽ tất cả metrics
evaluator.plot_all_scores(
    save_path="reports/figures/all_metrics_compare.png"
)

# 7. Tìm mô hình tốt nhất
best_model = evaluator.get_best_model(metric="silhouette")
print(f"Best model: {best_model['model']}")

# 8. Xóa kết quả nếu muốn reset
evaluator.clear_results()
"""
