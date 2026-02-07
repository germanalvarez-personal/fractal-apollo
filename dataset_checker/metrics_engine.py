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
            
            # Truncated: touching edges with configurable margin
            (
                (pl.col("x_min") <= self.config.truncation_margin) | (pl.col("x_max") >= (1.0 - self.config.truncation_margin)) |
                (pl.col("y_min") <= self.config.truncation_margin) | (pl.col("y_max") >= (1.0 - self.config.truncation_margin))
            ).alias("is_touching_border"),
        ])
            
        # --- IQR-Based Outlier Detection (Class-Relative) ---
        q = q.with_columns([
            pl.col("area_rel").quantile(0.25).over("class_id").alias("area_q1"),
            pl.col("area_rel").quantile(0.75).over("class_id").alias("area_q3"),
            pl.count("area_rel").over("class_id").alias("cls_count")
        ])
        
        q = q.with_columns(
            (pl.col("area_q3") - pl.col("area_q1")).alias("area_iqr")
        )
        
        q = q.with_columns([
            (pl.col("area_q1") - (self.config.area_iqr_low * pl.col("area_iqr"))).alias("area_lower"),
            (pl.col("area_q3") + (self.config.area_iqr_high * pl.col("area_iqr"))).alias("area_upper")
        ])

        # Refined Truncation Logic: 
        # Only flag as 'is_truncated' (outlier) if it touches border AND is statistically small.
        # Threshold is configurable via truncation_quantile (default 0.25/Q1).
        
        q = q.with_columns(
            pl.col("area_rel").quantile(self.config.truncation_quantile).over("class_id").alias("truncation_thresh")
        )
        
        q = q.with_columns(
            (pl.col("is_touching_border") & (pl.col("area_rel") < pl.col("truncation_thresh"))).alias("is_truncated")
        )
        
        # --- is_tiny Logic (Stats OR Fallback) ---
        # Strategy:
        # 1. If cls_count >= min_samples: Use IQR statistical lower bound.
        # 2. Else: Use absolute tiny_object_area (fallback).
        
        # Note: If tiny_object_area is None/0, fallback is effectively disabled (nothing is tiny)
        fallback_tiny = self.config.tiny_object_area if self.config.tiny_object_area is not None else -1.0
        
        tiny_cond_stats = (pl.col("cls_count") >= self.config.z_score_sample_min) & (pl.col("area_rel") < pl.col("area_lower"))
        tiny_cond_abs   = (pl.col("cls_count") < self.config.z_score_sample_min) & (pl.col("area_rel") < fallback_tiny)
        
        tiny_expr = tiny_cond_stats | tiny_cond_abs

        # --- is_oversized Logic (Stats OR Fallback) ---
        # 1. If cls_count >= min_samples: Use IQR statistical upper bound.
        # 2. Else: Use absolute oversized_safety_floor (fallback).
        
        oversized_cond_stats = (pl.col("cls_count") >= self.config.z_score_sample_min) & (pl.col("area_rel") > pl.col("area_upper"))
        oversized_cond_abs   = (pl.col("cls_count") < self.config.z_score_sample_min) & (pl.col("area_rel") > self.config.oversized_safety_floor)
        
        oversized_expr = oversized_cond_stats | oversized_cond_abs


        q = q.with_columns([
            tiny_expr.alias("is_tiny"),
            oversized_expr.alias("is_oversized"),
            (
                (pl.col("x_center") < 0) | (pl.col("x_center") > 1) |
                (pl.col("y_center") < 0) | (pl.col("y_center") > 1)
            ).alias("is_out_of_bounds"),
            
            # Exact Duplicate
            (pl.count("class_id").over(["file_path", "x_center", "y_center", "width", "height"]) > 1).alias("is_duplicate_exact"),
        ])
        
        # --- IoU-Based Duplicate Detection ---
        # Materialize here to prevent excessive graph complexity or re-scanning during self-join
        q = q.collect().lazy()
        
        # To do this safely, we need a unique ID per object.
        q = q.with_row_index("obj_id")
        
        # We process duplicates by looking for high overlap.
        # Since this is expensive, maybe we only run it if needed? but `lazy` is fine.
        
        # Self-join to find pairs (a, b) in same file, same class, where a.id < b.id
        # Then calc IoU.
        
        # Define IoU calculation expression
        # IoU = Inter / Union
        # Inter = max(0, min(max_x) - max(min_x)) * ...
        
        # We can't easily do this purely in lazy expressions without a join.
        # Let's verify if we want to add this complexity. Yes, user requested it.
        
        # We'll defer the IoU execution to `compute_metrics` assuming `q` is materialized or handled carefully.
        # Actually, let's append a boolean column `is_duplicate_iou` via a mapped function or join.
        
        # Strategy:
        # 1. Create a lightweight dataframe of coords.
        # 2. Join.
        # 3. Calculate.
        # 4. Filter > threshold.
        # 5. Get list of bad IDs.
        # 6. Flag those IDs in main df.
        
        # Since `q` is a LazyFrame, we can chain this.
        
        coords = q.select(["obj_id", "file_path", "class_id", "x_min", "x_max", "y_min", "y_max"])
        
        pairs = coords.join(coords, on=["file_path", "class_id"], suffix="_2")
        pairs = pairs.filter(pl.col("obj_id") < pl.col("obj_id_2"))
        
        # Calculate Intersection
        ix_min = pl.max_horizontal("x_min", "x_min_2")
        ix_max = pl.min_horizontal("x_max", "x_max_2")
        iy_min = pl.max_horizontal("y_min", "y_min_2")
        iy_max = pl.min_horizontal("y_max", "y_max_2")
        
        iw = (ix_max - ix_min).clip(0, 100) # clip 0 lower
        ih = (iy_max - iy_min).clip(0, 100)
        inter = iw * ih
        
        # Calculate Union
        # area1 = (x_max - x_min) * ...
        w1 = pl.col("x_max") - pl.col("x_min")
        h1 = pl.col("y_max") - pl.col("y_min")
        a1 = w1 * h1
        
        w2 = pl.col("x_max_2") - pl.col("x_min_2")
        h2 = pl.col("y_max_2") - pl.col("y_min_2")
        a2 = w2 * h2
        
        union = a1 + a2 - inter
        iou = inter / union
        
        # Filter High IoU
        high_iou_pairs = pairs.filter(iou > self.config.duplicate_iou_threshold).select(["obj_id", "obj_id_2"])
        
        # We need to flag BOTH obj_id and obj_id_2 as duplicates
        # concat
        # We rename to obj_id to use simple join
        dupe_df = pl.concat([
            high_iou_pairs.select(pl.col("obj_id")),
            high_iou_pairs.select(pl.col("obj_id_2").alias("obj_id"))
        ]).unique().collect()
        
        dupe_df = dupe_df.with_columns(pl.lit(True).alias("is_dupe_iou"))
        
        dupe_lazy = dupe_df.lazy()
        
        # Now join back to flag
        # Use simple join on obj_id
        q = q.join(dupe_lazy, on="obj_id", how="left")
        
        q = q.with_columns(
            (pl.col("is_dupe_iou").fill_null(False) | pl.col("is_duplicate_exact")).alias("is_duplicate")
        ).drop(["is_dupe_iou", "obj_id"]) # Cleanup

        # Z-Score Stats Validation
        # (Re-calculate stats over class_id including the possibly collected info?)
        # We already calc stats above for Area. Now for Aspect Ratio.
        
        # Note: We calculated cls_count above. Reuse it?
        # Use window function again to be safe/consistent
        q = q.with_columns([
            pl.mean("aspect_ratio").over("class_id").alias("ar_mean"),
            pl.std("aspect_ratio").over("class_id").alias("ar_std"),
            # cls_count already exists
        ])
        
        # Compute Z-score and is_stretched logic
        # Logic: if cls_count > z_score_sample_min: use z-score > mean
        #        else: use absolute bounds
        
        z_score_expr = ((pl.col("aspect_ratio") - pl.col("ar_mean")) / pl.col("ar_std")).abs()
        
        is_stretched_z = (
            (pl.col("cls_count") > self.config.z_score_sample_min) & 
            (z_score_expr > self.config.aspect_ratio_z_threshold) &
            (~pl.col("is_truncated")) # User feedback: Stretched implies valid full object, not cut off
        )
        is_stretched_abs = (
            (pl.col("cls_count") <= self.config.z_score_sample_min) & 
            ((pl.col("aspect_ratio") > self.config.aspect_ratio_abs_max) | (pl.col("aspect_ratio") < self.config.aspect_ratio_abs_min)) &
            (~pl.col("is_truncated"))
        )
        
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
