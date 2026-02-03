import polars as pl
import numpy as np
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
            # Calculate coordinates for truncation check
            (pl.col("x_center") - pl.col("width") / 2).alias("x_min"),
            (pl.col("x_center") + pl.col("width") / 2).alias("x_max"),
            (pl.col("y_center") - pl.col("height") / 2).alias("y_min"),
            (pl.col("y_center") + pl.col("height") / 2).alias("y_max"),
            
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
            ((pl.col("width") <= 0) | (pl.col("height") <= 0)).alias("is_inverted"),
            
            # Truncated: touching edges (with 1% margin as requested)
            (
                (pl.col("x_min") <= 0.01) | (pl.col("x_max") >= 0.99) |
                (pl.col("y_min") <= 0.01) | (pl.col("y_max") >= 0.99)
            ).alias("is_truncated"),
        ])
            
        # --- IQR-Based Outlier Detection (Class-Relative) ---
        # Calculate Q1, Q3, IQR per class for 'area_rel'
        q = q.with_columns([
            pl.col("area_rel").quantile(0.25).over("class_id").alias("area_q1"),
            pl.col("area_rel").quantile(0.75).over("class_id").alias("area_q3")
        ])
        
        q = q.with_columns(
            (pl.col("area_q3") - pl.col("area_q1")).alias("area_iqr")
        )
        
        # Define bounds
        # Lower bound = Q1 - (k_low * IQR)
        # Upper bound = Q3 + (k_high * IQR)
        q = q.with_columns([
            (pl.col("area_q1") - (self.config.area_iqr_low * pl.col("area_iqr"))).alias("area_lower"),
            (pl.col("area_q3") + (self.config.area_iqr_high * pl.col("area_iqr"))).alias("area_upper")
        ])
        
        # is_tiny:
        # 1. Statistically small (area < area_lower)
        # 2. OR Absolute safety floor (area < tiny_object_area) - captures extremely small noise regardless of distribution
        tiny_expr = (pl.col("area_rel") < pl.col("area_lower")) | (pl.col("area_rel") < self.config.tiny_object_area)
        
        # is_oversized:
        # 1. Statistically large (area > area_upper)
        # 2. OR Absolute ceiling (area > oversized_safety_floor)
        oversized_expr = (pl.col("area_rel") > pl.col("area_upper")) | (pl.col("area_rel") > self.config.oversized_safety_floor)


        q = q.with_columns([
            tiny_expr.alias("is_tiny"),
            oversized_expr.alias("is_oversized"),
            (
                (pl.col("x_center") < 0) | (pl.col("x_center") > 1) |
                (pl.col("y_center") < 0) | (pl.col("y_center") > 1)
            ).alias("is_out_of_bounds"),
            
            # Duplicate (exact) - we will add approximate overlap next
            (pl.count("class_id").over(["file_path", "x_center", "y_center", "width", "height"]) > 1).alias("is_duplicate_exact"),
        ])

        # Z-Score Stats Validation
        # We need to compute stats per class. 
        # Since this is lazy, we can use window functions.
        q = q.with_columns([
            pl.mean("aspect_ratio").over("class_id").alias("ar_mean"),
            pl.std("aspect_ratio").over("class_id").alias("ar_std"),
            pl.count("aspect_ratio").over("class_id").alias("cls_count")
        ])
        
        # Compute Z-score and is_stretched logic
        # Logic: if cls_count > 15: use z-score > 3
        #        else: use absolute bounds (> 5.0 or < 0.2)
        
        z_score_expr = ((pl.col("aspect_ratio") - pl.col("ar_mean")) / pl.col("ar_std")).abs()
        
        is_stretched_z = (pl.col("cls_count") > 15) & (z_score_expr > self.config.aspect_ratio_z_threshold)
        is_stretched_abs = (pl.col("cls_count") <= 15) & ((pl.col("aspect_ratio") > 5.0) | (pl.col("aspect_ratio") < 0.2))
        
        q = q.with_columns(
            (is_stretched_z | is_stretched_abs).alias("is_stretched")
        )
        
        # Collect results
        print("Executing lazy pipeline...")
        df = q.collect()
        
        # Post-process Overlap (Approximate Duplicate via IoU)
        df = self.check_overlap(df)
        
        # Combine duplicates
        df = df.with_columns(
            (pl.col("is_duplicate_exact") | pl.col("is_high_overlap")).alias("is_duplicate")
        )
        
        print(f"Processed {df.height} objects.")
        return df

    def check_overlap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Check for high overlap (IoU > threshold) between objects of the SAME class in the SAME image.
        This detects 'double labeling' errors.
        """
        # Optimized Polars Implementation (Rust-backed)
        # Avoids Python loops and numpy materialization
        
        # 1. Add index for tracking
        df_idx = df.with_row_index("idx")
        
        # 2. Filter candidates (files with > 1 object of same class)
        # We calculate count using window function if not already present, but it is (object_count is per file, we need per class/file)
        # Actually object_count in compute_metrics is count("class_id").over("file_path") which counts ALL objects in file.
        # We need specific class counts.
        
        candidates = df_idx.filter(pl.count("idx").over(["file_path", "class_id"]) > 1)
        
        if candidates.is_empty():
             return df.with_columns(pl.lit(False).alias("is_high_overlap"))
             
        # 3. Self-Join to create pairs
        # We only need coordinate columns + index
        cols = ["idx", "file_path", "class_id", "x_min", "y_min", "x_max", "y_max"]
        
        pairs = candidates.select(cols).join(
            candidates.select(cols),
            on=["file_path", "class_id"],
            suffix="_right"
        )
        
        # 4. Filter strictly upper triangle (idx < idx_right) to avoid self-match and duplicates
        pairs = pairs.filter(pl.col("idx") < pl.col("idx_right"))
        
        if pairs.is_empty():
             return df.with_columns(pl.lit(False).alias("is_high_overlap"))

        # 5. Calculate IoU using Expressions
        # Intersection
        ix1 = pl.max_horizontal("x_min", "x_min_right")
        iy1 = pl.max_horizontal("y_min", "y_min_right")
        ix2 = pl.min_horizontal("x_max", "x_max_right")
        iy2 = pl.min_horizontal("y_max", "y_max_right")
        
        inter_w = (ix2 - ix1).clip(0)
        inter_h = (iy2 - iy1).clip(0)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = (pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))
        area2 = (pl.col("x_max_right") - pl.col("x_min_right")) * (pl.col("y_max_right") - pl.col("y_min_right"))
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        # 6. Filter by threshold
        high_overlap_pairs = pairs.filter(iou > self.config.iou_duplicate_threshold)
        
        # 7. Collect indices to flag
        # We flag BOTH objects in the pair
        bad_indices_left = high_overlap_pairs["idx"]
        bad_indices_right = high_overlap_pairs["idx_right"]
        
        all_bad_indices = pl.concat([bad_indices_left, bad_indices_right]).unique()
        
        # 8. Update original DF
        # We can use is_in
        return df_idx.with_columns(
            pl.col("idx").is_in(all_bad_indices).alias("is_high_overlap")
        ).drop("idx")
