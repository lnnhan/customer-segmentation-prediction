"""
Main Script - Highlands Coffee Customer Segmentation

Pipeline hoàn chỉnh từ preprocessing đến clustering và visualization
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.trainer import ModelTrainer, TrainingConfig
from src.models.evaluator import ClusteringEvaluator
from src.models.tuning import HyperparameterTuner, TuningConfig
from src.utils import setup_logger


def train_single_model(logger):
    """
    Mode 1: Train một model cụ thể với cấu hình cố định
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 1: TRAIN SINGLE MODEL")
    logger.info("="*80)
    
    # Cấu hình model
    config = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="kmeans",  
        n_clusters=5,
        model_params={
            "n_init": 20,
            "max_iter": 500
        },
        model_path="results/kmeans_model.pkl"
    )
    
    # Train
    evaluator = ClusteringEvaluator()
    trainer = ModelTrainer(config=config, evaluator=evaluator)
    
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    
    # Log metrics
    logger.info("\nTRAINING METRICS:")
    logger.info(f"   Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}")
    logger.info(f"   Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.2f}")
    logger.info(f"   Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.4f}")
    logger.info(f"   Number of Clusters: {metrics.get('n_clusters', 'N/A')}")
    
    # Lưu kết quả
    trainer.save_model()
    trainer.save_labels("results/kmeans_labels.csv")
    
    logger.info("\n✅ Single model training completed!")
    logger.info(f"   Model saved: {config.model_path}")
    logger.info(f"   Labels saved: results/kmeans_labels.csv")
    
    return trainer, metrics


def hyperparameter_tuning(logger):
    """
    Mode 2: Grid search tất cả models để tìm best hyperparameters
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 2: HYPERPARAMETER TUNING")
    logger.info("="*80)
    
    tuning_config = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/tuning_results.csv",
        metric_selection="composite",
        silhouette_weight=0.4,
        calinski_weight=0.3,
        davies_weight=0.3
    )
    
    evaluator = ClusteringEvaluator()
    tuner = HyperparameterTuner(config=tuning_config, evaluator=evaluator)
    
    # Run grid search cho tất cả models
    tuner.run_all_models()
    
    # Lưu kết quả
    tuner.save_results()
    tuner.save_best_model_and_df(
        model_path="results/best_model.pkl",
        df_path="results/clustered_data.csv"
    )
    
    # Log best results
    summary = tuner.get_summary()
    if not summary.empty:
        best = summary.iloc[0]
        logger.info("\n🏆 BEST MODEL FOUND:")
        logger.info(f"   Model Type: {best.get('model_type', 'N/A')}")
        logger.info(f"   N Clusters: {best.get('n_clusters', 'N/A')}")
        logger.info(f"   Silhouette: {best.get('silhouette', 0):.4f}")
        logger.info(f"   Calinski-Harabasz: {best.get('calinski_harabasz', 0):.2f}")
        logger.info(f"   Davies-Bouldin: {best.get('davies_bouldin', 0):.4f}")
        if 'composite_score' in best:
            logger.info(f"   Composite Score: {best.get('composite_score', 0):.4f}")
    
    logger.info("\n✅ Hyperparameter tuning completed!")
    logger.info(f"   Results saved: {tuning_config.results_path}")
    logger.info(f"   Best model: results/best_model.pkl")
    logger.info(f"   Clustered data: results/clustered_data.csv")
    
    return tuner


def compare_models(logger):
    """
    Mode 3: So sánh nhanh 4 models với cấu hình mặc định
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 3: QUICK MODEL COMPARISON")
    logger.info("="*80)
    
    evaluator = ClusteringEvaluator()
    results = []
    
    # 1. KMeans
    logger.info("\n[1/4] Testing KMeans...")
    config_kmeans = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="kmeans",
        n_clusters=5,
        model_params={"n_init": 20}
    )
    trainer = ModelTrainer(config_kmeans, evaluator)
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    results.append({"model": "KMeans", **metrics})
    
    # 2. GMM
    logger.info("\n[2/4] Testing GMM...")
    config_gmm = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="gmm",
        n_clusters=5,
        model_params={"covariance_type": "full"}
    )
    trainer = ModelTrainer(config_gmm, evaluator)
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    results.append({"model": "GMM", **metrics})
    
    # 3. DBSCAN
    logger.info("\n[3/4] Testing DBSCAN...")
    config_dbscan = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="dbscan",
        model_params={"eps": 2.0, "min_samples": 10}
    )
    trainer = ModelTrainer(config_dbscan, evaluator)
    trainer.load_data()
    trainer.train_model()
    
    labels = trainer.get_cluster_labels()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters >= 2:
        metrics = trainer.evaluate()
        results.append({"model": "DBSCAN", **metrics})
    else:
        logger.warning(f"  ⚠ DBSCAN only found {n_clusters} cluster(s), skipping evaluation")
    
    # 4. HDBSCAN
    logger.info("\n[4/4] Testing HDBSCAN...")
    try:
        config_hdbscan = TrainingConfig(
            data_path="data/processed/encoded_data.csv",
            model_type="hdbscan",
            model_params={"min_cluster_size": 15, "min_samples": 10}
        )
        trainer = ModelTrainer(config_hdbscan, evaluator)
        trainer.load_data()
        trainer.train_model()
        
        labels = trainer.get_cluster_labels()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            metrics = trainer.evaluate()
            results.append({"model": "HDBSCAN", **metrics})
        else:
            logger.warning(f"  ⚠ HDBSCAN only found {n_clusters} cluster(s), skipping evaluation")
    except ImportError:
        logger.warning("  ⚠ HDBSCAN not installed, skipping...")
    
    # Hiển thị bảng so sánh
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL COMPARISON RESULTS")
    logger.info("="*80)
    logger.info("\n" + df_results.to_string(index=False))
    
    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/model_comparison.csv", index=False)
    logger.info(f"\n💾 Comparison saved: results/model_comparison.csv")
    
    return df_results


def compare_models_with_tuning(logger):
    """
    Mode 5: Chạy hyperparameter tuning cho cả 4 models và so sánh composite scores
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 5: FULL TUNING & COMPARISON (COMPOSITE METRIC)")
    logger.info("="*80)
    
    evaluator = ClusteringEvaluator()
    all_results = []
    
    # 1. KMeans Tuning
    logger.info("\n[1/4] Tuning KMeans...")
    config_kmeans = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/kmeans_composite_tuning.csv",
        metric_selection="composite"
    )
    tuner_kmeans = HyperparameterTuner(config=config_kmeans, evaluator=evaluator)
    tuner_kmeans.load_data()
    tuner_kmeans.run_grid_search("kmeans", {
        "n_clusters": [3, 4, 5, 6, 7, 8, 9, 10],
        "init": ["k-means++"],
        "n_init": [10, 20],
        "max_iter": [300]
    })
    tuner_kmeans.save_results()
    summary_kmeans = tuner_kmeans.get_summary()
    if not summary_kmeans.empty:
        best_kmeans = summary_kmeans.iloc[0]
        all_results.append({
            "model": "KMeans",
            "n_clusters": best_kmeans.get("n_clusters", "N/A"),
            "silhouette": best_kmeans.get("silhouette", 0),
            "calinski_harabasz": best_kmeans.get("calinski_harabasz", 0),
            "davies_bouldin": best_kmeans.get("davies_bouldin", 0),
            "composite_score": best_kmeans.get("composite_score", 0)
        })
    
    # 2. GMM Tuning
    logger.info("\n[2/4] Tuning GMM...")
    config_gmm = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/gmm_composite_tuning.csv",
        metric_selection="composite"
    )
    tuner_gmm = HyperparameterTuner(config=config_gmm, evaluator=evaluator)
    tuner_gmm.load_data()
    tuner_gmm.run_grid_search("gmm", {
        "n_clusters": [3, 4, 5, 6, 7, 8],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "n_init": [10],
        "max_iter": [200]
    })
    tuner_gmm.save_results()
    summary_gmm = tuner_gmm.get_summary()
    if not summary_gmm.empty:
        best_gmm = summary_gmm.iloc[0]
        all_results.append({
            "model": "GMM",
            "n_clusters": best_gmm.get("n_clusters", "N/A"),
            "silhouette": best_gmm.get("silhouette", 0),
            "calinski_harabasz": best_gmm.get("calinski_harabasz", 0),
            "davies_bouldin": best_gmm.get("davies_bouldin", 0),
            "composite_score": best_gmm.get("composite_score", 0)
        })
    
    # 3. DBSCAN Tuning
    logger.info("\n[3/4] Tuning DBSCAN...")
    config_dbscan = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/dbscan_composite_tuning.csv",
        metric_selection="composite"
    )
    tuner_dbscan = HyperparameterTuner(config=config_dbscan, evaluator=evaluator)
    tuner_dbscan.load_data()
    tuner_dbscan.run_grid_search("dbscan", {
        "eps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "min_samples": [5, 10, 15, 20],
        "metric": ["euclidean", "manhattan"]
    })
    tuner_dbscan.save_results()
    summary_dbscan = tuner_dbscan.get_summary()
    if not summary_dbscan.empty:
        best_dbscan = summary_dbscan.iloc[0]
        all_results.append({
            "model": "DBSCAN",
            "n_clusters": best_dbscan.get("n_clusters", "N/A"),
            "silhouette": best_dbscan.get("silhouette", 0),
            "calinski_harabasz": best_dbscan.get("calinski_harabasz", 0),
            "davies_bouldin": best_dbscan.get("davies_bouldin", 0),
            "composite_score": best_dbscan.get("composite_score", 0)
        })
    
    # 4. HDBSCAN Tuning
    logger.info("\n[4/4] Tuning HDBSCAN...")
    try:
        config_hdbscan = TuningConfig(
            data_path="data/processed/encoded_data.csv",
            results_path="results/hdbscan_composite_tuning.csv",
            metric_selection="composite"
        )
        tuner_hdbscan = HyperparameterTuner(config=config_hdbscan, evaluator=evaluator)
        tuner_hdbscan.load_data()
        tuner_hdbscan.run_grid_search("hdbscan", {
            "min_cluster_size": [10, 20, 30, 50, 100],
            "min_samples": [10, 20],
            "metric": ["euclidean", "manhattan"],
            "cluster_selection_method": ["eom"]
        })
        tuner_hdbscan.save_results()
        summary_hdbscan = tuner_hdbscan.get_summary()
        if not summary_hdbscan.empty:
            best_hdbscan = summary_hdbscan.iloc[0]
            all_results.append({
                "model": "HDBSCAN",
                "n_clusters": best_hdbscan.get("n_clusters", "N/A"),
                "silhouette": best_hdbscan.get("silhouette", 0),
                "calinski_harabasz": best_hdbscan.get("calinski_harabasz", 0),
                "davies_bouldin": best_hdbscan.get("davies_bouldin", 0),
                "composite_score": best_hdbscan.get("composite_score", 0)
            })
    except ImportError:
        logger.warning("  ⚠ HDBSCAN not installed, skipping...")
    
    # Hiển thị bảng so sánh
    df_comparison = pd.DataFrame(all_results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 TUNING COMPARISON RESULTS (COMPOSITE METRIC)")
    logger.info("="*80)
    logger.info("\n" + df_comparison.to_string(index=False))
    
    # Tìm model tốt nhất dựa trên composite_score
    if not df_comparison.empty:
        best_idx = df_comparison['composite_score'].idxmax()
        best_model_name = df_comparison.loc[best_idx, 'model']
        best_composite = df_comparison.loc[best_idx, 'composite_score']
        
        logger.info(f"\n🏆 Best model: {best_model_name} (Composite Score: {best_composite:.4f})")
        
        # Lưu best model và clustered data
        if best_model_name == "KMeans":
            best_tuner = tuner_kmeans
        elif best_model_name == "GMM":
            best_tuner = tuner_gmm
        elif best_model_name == "DBSCAN":
            best_tuner = tuner_dbscan
        elif best_model_name == "HDBSCAN":
            best_tuner = tuner_hdbscan
        
        best_tuner.save_best_model_and_df(
            model_path="results/best_model_composite.pkl",
            df_path="results/clustered_data_composite.csv"
        )
        
        logger.info(f"   Best model saved: results/best_model_composite.pkl")
        logger.info(f"   Clustered data saved: results/clustered_data_composite.csv")
    
    # Lưu kết quả so sánh
    os.makedirs("results", exist_ok=True)
    df_comparison.to_csv("results/model_comparison_composite.csv", index=False)
    logger.info(f"\n💾 Comparison saved: results/model_comparison_composite.csv")
    
    return df_comparison


def main():
    """Main function với menu chọn mode"""
    logger = setup_logger("coffee_project_main", log_dir="logs")
    
    logger.info("\n" + "="*80)
    logger.info("HIGHLANDS COFFEE CUSTOMER SEGMENTATION")
    logger.info("="*80)
    logger.info("\nChọn mode:")
    logger.info("  1. Train single model (nhanh, test model cụ thể)")
    logger.info("  2. Hyperparameter tuning (chậm, tìm best config)")
    logger.info("  3. Quick model comparison (so sánh 4 models)")
    logger.info("  4. Run all (chạy tất cả)")
    logger.info("  5. Full tuning + comparison (composite metric)")
    
    choice = input("\nNhập lựa chọn (1/2/3/4/5): ").strip()
    
    if choice == "1":
        train_single_model(logger)
    
    elif choice == "2":
        hyperparameter_tuning(logger)
    
    elif choice == "3":
        compare_models(logger)
    
    elif choice == "4":
        logger.info("\nRunning all modes...")
        train_single_model(logger)
        compare_models(logger)
        hyperparameter_tuning(logger)
    
    elif choice == "5":
        compare_models_with_tuning(logger)
    
    else:
        logger.error("Invalid choice!")
        return
    
    logger.info("\n" + "="*80)
    logger.info("ALL DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
