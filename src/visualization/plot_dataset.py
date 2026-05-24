# Dataset plots - EDA charts
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from src.config import FIGURES_DIR
except:
    # fallback nếu chạy độc lập
    FIGURES_DIR = Path("reports/figures")


# ============================
# 1. Class Loader
# ============================

class DataLoader:
    def __init__(self, raw_folder=None):
        if raw_folder is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            raw_folder = os.path.join(project_root, "data", "raw")
        self.raw_folder = Path(raw_folder)
        self.datasets = {}

    def load_all_csv(self):
        csv_files = list(self.raw_folder.glob("*.csv"))

        for file in csv_files:
            try:
                # Tự động phát hiện mã hóa
                raw_bytes = file.read_bytes()[:200000]
                enc = "utf-8"
                try:
                    raw_bytes.decode(enc)
                except UnicodeDecodeError:
                    enc = "latin-1"

                try:
                    df = pd.read_csv(
                        file,
                        encoding=enc,
                        sep=None,          # tự động phát hiện
                        engine="python",
                    )
                except Exception:
                    # Fallback: skip bad lines
                    df = pd.read_csv(
                        file,
                        encoding=enc,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"   
                    )
                    print(f" {file.name}: có dòng lỗi → đã skip")

                self.datasets[file.name] = df
                print(f"✓ Loaded: {file.name} | shape={df.shape} | enc={enc}")

            except Exception as e:
                print(f"✗ Lỗi load {file.name}: {e}")

        return self.datasets


# ============================
# 2. Class PlotDataset
# ============================

class PlotDataset:

    def __init__(self, output_dir: str | None = None):
        if output_dir is None:
            output_dir = str(FIGURES_DIR)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Save figure ----
    def _save_fig(self, path):
        plt.tight_layout()
        plt.savefig(path,dpi = 200)
        plt.close()

    # ---- Histogram ----
    def hist(self, df, col, out_folder):
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        self._save_fig(os.path.join(out_folder, f"hist_{col}.png"))

    # ---- Boxplot ----
    def boxplot(self, df, col, out_folder):
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        self._save_fig(os.path.join(out_folder, f"box_{col}.png"))

    # ---- Count plot for categorical ----
    def countplot(self, df, col, out_folder):
        plt.figure(figsize=(8, 4))
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        self._save_fig(os.path.join(out_folder, f"count_{col}.png"))

    # ---- Combined plots: create a single figure containing one subplot per column ----
    def combined_boxplots(self, df, cols, out_folder, max_cols=12, suffix="all"):
        cols = list(cols)[:max_cols]
        n = len(cols)
        if n == 0:
            print(f"No categorical columns with ≤50 categories")
            return
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(cols):
            ax = axes[i]
            # Nếu cột là numeric, thì vẽ biểu đồ boxplot; nếu là categorical, biểu đồ boxplot đếm giá trị
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f"Boxplot - {col}")
                else:
                    counts = df[col].astype(str).value_counts()
                    # boxplot of counts distribution (one box summarizing category frequencies)
                    sns.boxplot(x=counts, ax=ax)
                    ax.set_title(f"Boxplot(Counts) - {col}")
                    ax.set_xlabel("Counts")
            except Exception:
                ax.set_visible(False)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        self._save_fig(os.path.join(out_folder, f"boxplots_{suffix}.png"))

    def combined_histograms(self, df, cols, out_folder, max_cols=12, suffix="all"):
        cols = list(cols)[:max_cols]
        n = len(cols)
        if n == 0:
            print(f"No categorical columns with ≤50 categories")
            return
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(cols):
            ax = axes[i]
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f"Hist - {col}")
                else:
                    # Xoay ngang biểu đồ
                    vc = df[col].astype(str).value_counts()
                    vc.plot(kind='barh', ax=ax)
                    ax.set_title(f"Count - {col}")
                    ax.set_xlabel("Count")
                    ax.set_ylabel(col)
            except Exception:
                ax.set_visible(False)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        self._save_fig(os.path.join(out_folder, f"histograms_{suffix}.png"))

    def combined_countplots(self, df, cols, out_folder, max_cols=12, suffix="all"):
        cols = [c for c in list(cols) if df[c].nunique() <= 50][:max_cols]
        n = len(cols)
        if n == 0:
            return
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3.5))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(cols):
            ax = axes[i]
            # dùng value_counts cho categorical bar plot
            vc = df[col].astype(str).value_counts()
            vc.plot(kind='bar', ax=ax)
            ax.set_title(f"Count - {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        self._save_fig(os.path.join(out_folder, f"countplots_{suffix}.png"))

    # ---- Correlation ----
    def correlation(self, df, out_folder):
        plt.figure(figsize=(8, 6))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        self._save_fig(os.path.join(out_folder, f"correlation.png"))

    # ---- Generate all plots ----
    def generate_plots(self, datasets):
        """
        datasets: dict gồm {filename: dataframe}
        """

        for filename, df in datasets.items():
            dataset_name = filename.replace(".csv", "")
            out_folder = os.path.join(str(self.output_dir), dataset_name)
            Path(out_folder).mkdir(parents=True, exist_ok=True)

            print(f"\n📊 Đang vẽ: {filename}")

            numeric_cols = list(df.select_dtypes(include="number").columns)
            categorical_cols = list(df.select_dtypes(include=["object"]).columns)
            print(f"   Shape: {df.shape}")

            # Biểu đồ numeric categorical vẽ riêng
            print(f"   Numeric columns: {numeric_cols}")
            print(f"   Categorical columns: {categorical_cols}")

            if len(numeric_cols) > 0:
                # Numeric figures
                self.combined_histograms(df, numeric_cols, out_folder, suffix="numeric")
                self.combined_boxplots(df, numeric_cols, out_folder, suffix="numeric")
                if len(numeric_cols) > 1:
                    self.correlation(df, out_folder)

            if len(categorical_cols) > 0:
                # Categorical figures (horizontal bar charts for histograms)
                self.combined_histograms(df, categorical_cols, out_folder, suffix="categorical")
                self.combined_boxplots(df, categorical_cols, out_folder, suffix="categorical")
            else:
                print("Không còn cột nào để vẽ")

            print(f"   ✔ Lưu ảnh vào: {os.path.join(self.output_dir, dataset_name)}/")


# ============================
# 3. Run trực tiếp
# ============================

if __name__ == "__main__":
    loader = DataLoader()
    datasets = loader.load_all_csv()

    plotter = PlotDataset() 
    plotter.generate_plots(datasets)

    print("\nHoàn tất vẽ biểu đồ cho toàn bộ dataset!")
