import polars as pl
from .config import DatasetConfig

class MetricsEngine:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def compute_metrics(self, lazy_df: pl.LazyFrame) -> pl.DataFrame:
        """
        Lazy evaluation of geometric stats and quality flags.
        """
        # Geometric Stats
        q = lazy_df.with_columns([
            (pl.col("width") * pl.col("height")).alias("area_rel"),
            (pl.col("width") / pl.col("height")).alias("aspect_ratio"),
            pl.min_horizontal(
                pl.col("x_center"),
                1.0 - pl.col("x_center"),
                pl.col("y_center"),
                1.0 - pl.col("y_center")
            ).alias("edge_prox")
        ])

        # New Metrics & Flags
        q = q.with_columns([
            pl.count("class_id").over("file_path").alias("object_count"),
            ((pl.col("width") <= 0) | (pl.col("height") <= 0)).alias("is_inverted")
        ])

        # Flags based on config
        q = q.with_columns([
            (pl.col("area_rel") < self.config.tiny_object_area).alias("is_tiny"),
            (
                (pl.col("x_center") < 0) | (pl.col("x_center") > 1) |
                (pl.col("y_center") < 0) | (pl.col("y_center") > 1)
            ).alias("is_out_of_bounds"),
            (
                (pl.col("aspect_ratio") > 5.0) | (pl.col("aspect_ratio") < 0.2)
            ).alias("is_stretched"),
            (pl.count("class_id").over(["file_path", "x_center", "y_center", "width", "height"]) > 1).alias("is_duplicate"),
        ])
        
        # Collect results
        print("Executing lazy pipeline...")
        df = q.collect()
        
        # Post-process Overlap
        df = self.check_overlap(df)
        
        print(f"Processed {df.height} objects.")
        return df

    def check_overlap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimized Overlap Check.
        Full N^2 IoU is expensive.
        We can approximate high overlap by checking for very close centers and similar dimensions.
        """
        # Placeholder for complex IoU. 
        # For now, default False.
        return df.with_columns(pl.lit(False).alias("is_high_overlap"))
