"""
Hyperparameter Tuning Module for Coffee Project
"""

import logging
import itertools
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Import các module dự án
from src.models.trainer import ModelTrainer, TrainingConfig
from src.models.evaluator import ClusteringEvaluator


@dataclass
class TuningConfig:
    data_path: str = "data/processed/encoded_data.csv"
    results_path: str = "results/tuning_results.csv"
    random_state: int = 42
    metric_selection: str = "composite"  # 'silhouette' | 'calinski_harabasz' | 'davies_bouldin' | 'composite'
    # Composite score weights (chỉ dùng khi metric_selection='composite')
    silhouette_weight: float = 0.4
    calinski_weight: float = 0.3
    davies_weight: float = 0.3


class HyperparameterTuner:
    def __init__(self, config: TuningConfig, evaluator: ClusteringEvaluator, logger: Optional[logging.Logger] = None):
        self.config = config
        self.evaluator = evaluator

        # Logger setup
        if logger is None:
            self.logger = logging.getLogger("HyperparameterTuner")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.df_data: Optional[pd.DataFrame] = None
        self.all_results: List[dict] = []

        # Best model tracking
        self.best_model: Optional[Any] = None
        self.best_labels: Optional[np.ndarray] = None
        self.best_params: Optional[dict] = None
        self.best_metric: Optional[float] = None

    def load_data(self):
        self.logger.info(f"📂 Loading data for tuning from {self.config.data_path}...")
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"File not found: {self.config.data_path}")
        self.df_data = pd.read_csv(self.config.data_path)
        self.logger.info(f"  ✓ Data loaded: {self.df_data.shape}")

    def _generate_param_grid(self, model_type: str, param_grid: Dict[str, List[Any]]):
        keys = param_grid.keys()
        values = param_grid.values()
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for c in combos:
            c['model_type'] = model_type
        return combos

    def run_grid_search(self, model_type: str, param_grid: Dict[str, List[Any]]):
        if self.df_data is None:
            self.load_data()

        self.logger.info(f"\nStarting Grid Search for: {model_type.upper()}")
        combos = self._generate_param_grid(model_type, param_grid)
        self.logger.info(f"  Total combinations to test: {len(combos)}")

        for i, params in enumerate(combos, 1):
            n_clusters = params.get("n_clusters", 5)
            model_params = {k: v for k, v in params.items() if k not in ["model_type", "n_clusters"]}

            param_str = "_".join([f"{k}{v}" for k, v in params.items() if k != "model_type"])
            model_name_id = f"{model_type.upper()}_{param_str}"

            try:
                cfg = TrainingConfig(
                    data_path=self.config.data_path,
                    model_type=model_type,
                    n_clusters=n_clusters,
                    random_state=self.config.random_state,
                    model_params=model_params,
                    model_path="temp.pkl"  # Không lưu model thực tế
                )

                trainer = ModelTrainer(cfg, self.evaluator, logger=logging.getLogger("Silent"))
                trainer.df = self.df_data
                trainer.X = self.df_data.values
                trainer.train_model()
                labels = trainer.get_cluster_labels()

                metrics = self.evaluator.evaluate_once(
                    X=trainer.X,
                    labels=labels,
                    model_name=model_name_id
                )

                record = {**params, **metrics}
                self.all_results.append(record)

                # Kiểm tra xem có phải model tốt nhất không
                # Note: Với composite score, sẽ tính sau khi có tất cả results
                # Nên ở đây tạm thời track theo metric_selection (hoặc silhouette)
                if self.config.metric_selection == "composite":
                    # Tạm dùng silhouette để track trong quá trình chạy
                    metric_val = metrics['silhouette']
                else:
                    metric_val = metrics[self.config.metric_selection]
                
                is_better = False
                if self.best_model is None:
                    is_better = True
                else:
                    if self.config.metric_selection == "davies_bouldin":
                        is_better = metric_val < self.best_metric
                    else:
                        is_better = metric_val > self.best_metric

                if is_better:
                    self.best_model = trainer.model
                    self.best_labels = labels
                    self.best_params = params
                    self.best_metric = metric_val

                self.logger.info(
                    f"  [{i}/{len(combos)}] {model_name_id}: "
                    f"Sil={metrics['silhouette']:.4f}, CH={metrics['calinski_harabasz']:.2f}, "
                    f"DB={metrics['davies_bouldin']:.4f}, Clusters={metrics['n_clusters']}"
                )

            except ValueError as e:
                self.logger.warning(f"  [{i}] {model_name_id}: ⚠️ Failed ({str(e)})")
            except Exception as e:
                self.logger.error(f"  [{i}] {model_name_id}: ❌ Error: {str(e)}")

    def run_all_models(self):
        self.run_grid_search("kmeans", {
            "n_clusters": [3, 4, 5, 6, 7, 8, 9, 10],
            "init": ["k-means++"],
            "n_init": [10, 20],
            "max_iter": [300]
        })
        self.run_grid_search("gmm", {
            "n_clusters": [3, 4, 5, 6, 7, 8],
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "n_init": [10],
            "max_iter": [200]
        })
        self.run_grid_search("dbscan", {
            "eps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "min_samples": [5, 10, 15, 20],
            "metric": ["euclidean", "manhattan"]
        })
        try:
            import hdbscan
            self.run_grid_search("hdbscan", {
                "min_cluster_size": [10, 20, 30, 50, 100],
                "min_samples": [10, 20],
                "metric": ["euclidean", "manhattan"],
                "cluster_selection_method": ["eom"]
            })
        except ImportError:
            self.logger.warning("HDBSCAN not installed, skipping...")

    def _calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Tính composite score từ 3 metrics:
        - Silhouette (0-1, higher better) → normalize giữ nguyên
        - Calinski-Harabasz (unbounded, higher better) → normalize 0-1 bằng min-max
        - Davies-Bouldin (unbounded, lower better) → invert và normalize
        
        Score cuối = weighted average của 3 metrics đã normalize
        """
        # Normalize Silhouette (đã ở 0-1)
        sil_norm = df['silhouette']
        
        # Normalize Calinski-Harabasz (min-max scaling)
        ch_min, ch_max = df['calinski_harabasz'].min(), df['calinski_harabasz'].max()
        if ch_max > ch_min:
            ch_norm = (df['calinski_harabasz'] - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = pd.Series([0.5] * len(df))
        
        # Normalize Davies-Bouldin (invert vì lower is better, then min-max)
        db_min, db_max = df['davies_bouldin'].min(), df['davies_bouldin'].max()
        if db_max > db_min:
            db_inverted = db_max - df['davies_bouldin']  # Invert
            db_norm = (db_inverted - (db_max - db_max)) / (db_max - db_min)
        else:
            db_norm = pd.Series([0.5] * len(df))
        
        # Weighted composite score
        composite = (
            self.config.silhouette_weight * sil_norm +
            self.config.calinski_weight * ch_norm +
            self.config.davies_weight * db_norm
        )
        
        return composite
    
    def get_summary(self) -> pd.DataFrame:
        if not self.all_results:
            return pd.DataFrame()
        df = pd.DataFrame(self.all_results)
        
        # Nếu dùng composite score
        if self.config.metric_selection == "composite":
            df['composite_score'] = self._calculate_composite_score(df)
            return df.sort_values(by='composite_score', ascending=False)
        
        # Nếu dùng single metric
        ascending = True if self.config.metric_selection == "davies_bouldin" else False
        return df.sort_values(by=self.config.metric_selection, ascending=ascending)

    def save_results(self):
        df = self.get_summary()
        if df.empty:
            self.logger.warning("No results to save.")
            return

        os.makedirs(os.path.dirname(self.config.results_path), exist_ok=True)
        df.to_csv(self.config.results_path, index=False)
        self.logger.info(f"\nTuning results saved to: {self.config.results_path}")

        self.logger.info("\n🏆 TOP 3 MODELS:")
        cols = ['model_type', 'n_clusters', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
        if 'composite_score' in df.columns:
            cols.append('composite_score')
        printable = [c for c in cols if c in df.columns]
        print(df[printable].head(3).to_string(index=False))
        
        # Nếu dùng composite, update best model theo composite score
        if self.config.metric_selection == "composite" and 'composite_score' in df.columns:
            best_row_idx = df.index[0]  # Đã sort rồi
            # Tìm lại model tương ứng từ all_results
            best_result = self.all_results[best_row_idx]
            self.best_params = {k: v for k, v in best_result.items() 
                              if k in ['model_type', 'n_clusters'] or k not in ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_clusters', 'model', 'composite_score']}
            self.logger.info(f"\nBest model updated based on composite score: {best_result.get('model_type', 'unknown')}")

    def save_best_model_and_df(self, model_path: str = "results/best_model.pkl", df_path: str = "results/clustered_data.csv"):
        if self.best_model is None or self.best_labels is None:
            self.logger.warning("No best model found to save.")
            return

        import pickle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)
        self.logger.info(f"Best model saved to: {model_path}")

        df_clustered = self.df_data.copy()
        df_clustered["cluster"] = self.best_labels
        df_clustered.to_csv(df_path, index=False)
        self.logger.info(f"Clustered DataFrame saved to: {df_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    tuning_cfg = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/hyperparameter_tuning_results.csv",
        metric_selection="silhouette"
    )

    evaluator = ClusteringEvaluator()
    tuner = HyperparameterTuner(tuning_cfg, evaluator)

    tuner.run_all_models()
    tuner.save_results()
    tuner.save_best_model_and_df(
        model_path="results/best_model.pkl",
        df_path="results/clustered_data.csv"
    )
