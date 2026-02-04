from pathlib import Path
from dataset_checker.config import DatasetConfig
from dataset_checker.data_loader import DataLoader
from dataset_checker.metrics_engine import MetricsEngine
from dataset_checker.visualizer import Visualizer
import time

class YoloRustAnalyzer:
    def __init__(self, dataset_path: str, output_dir: str = "outputs", img_ext: str = 'jpg', label_ext: str = 'txt', **kwargs):
        # Initialize Config
        self.config = DatasetConfig(
            dataset_path=dataset_path, 
            img_ext=img_ext, 
            label_ext=label_ext,
            **kwargs
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Modules
        self.loader = DataLoader(self.config)
        self.metrics = MetricsEngine(self.config)
        self.viz = Visualizer(self.config, self.loader)
        
        self.df = None

    def setup_coco(self, download_dir="inputs/coco_val2017"):
        # Ensure input dir exists if we are going to download there
        Path(download_dir).parent.mkdir(parents=True, exist_ok=True)
        self.loader.setup_coco(download_dir)

    def load_class_names(self, yaml_path):
        self.loader.load_class_names(Path(yaml_path))

    def analyze(self):
        """
        Main pipeline execution method.
        """
        t0 = time.time()
        print("--- Starting Analysis ---")
        
        # 1. Load Data
        lazy_df = self.loader.load_data()
        
        # 2. Compute Metrics
        self.df = self.metrics.compute_metrics(lazy_df)
        
        # 3. Validation
        self.loader.validate_dataset(self.df)
        
        print(f"Analysis complete in {time.time() - t0:.2f}s")
        return self.df

    def plot_dashboard(self, filename="dashboard.png"):
        if self.df is None:
            self.analyze()
        save_path = self.output_dir / filename
        self.viz.plot_dashboard(self.df, self.loader.total_images, str(save_path))

    def generate_stratified_mosaic(self, filename="stratified_mosaic.png"):
        if self.df is None:
            self.analyze()
        save_path = self.output_dir / filename
        self.viz.generate_stratified_mosaic(self.df, str(save_path))

    def generate_outlier_report(self, filename_prefix="outliers"):
        import json
        import polars as pl
        if self.df is None: self.analyze()
        
        # Define flags to report on
        flags = [
            "is_tiny", 
            "is_oversized", 
            "is_stretched", 
            "is_truncated", 
            "is_duplicate", 
            "is_inverted", 
            "is_out_of_bounds"
        ]
        
        for flag in flags:
            # Filter for specific outlier type
            flag_df = self.df.filter(self.df[flag])
            
            if flag_df.height == 0:
                continue
                
            report = {}
            # Group by class_id
            grouped = flag_df.group_by("class_id").agg(
                pl.col("file_path").unique().alias("paths")
            )
            
            for row in grouped.iter_rows(named=True):
                cid = row["class_id"]
                paths = row["paths"]
                cname = self.loader.class_names.get(cid, f"class_{cid}")
                report[cname] = paths
            
            # Create outliers directory
            outliers_dir = self.output_dir / "outliers"
            outliers_dir.mkdir(exist_ok=True)
            
            # Save separate file
            # e.g. outliers_is_tiny.json
            save_name = f"{filename_prefix}_{flag}.json"
            save_path = outliers_dir / save_name
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Saved {save_name} ({flag_df.height} objects)")
            
            # Generate Mosaic for this outlier type
            mosaic_name = f"mosaic_{flag}.png"
            mosaic_path = outliers_dir / mosaic_name
            self.viz.generate_outlier_mosaic(flag_df, flag, str(mosaic_path))

    # Backwards compatibility wrappers if needed
    def check_ghost_labels(self):
        if self.df is None: self.analyze()
        # Already run in analyze, but can re-run
        self.loader.validate_dataset(self.df)

    def generate_heatmap(self, **kwargs):
        if self.df is None: self.analyze()
        return self.viz.generate_heatmap(self.df, **kwargs)
