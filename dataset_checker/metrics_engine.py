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
            
            # Oversized
            (pl.col("area_rel") > self.config.oversized_object_area).alias("is_oversized")
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

        # Flags based on config
        # is_tiny: Check map first, logic is tricky in polars expr without UDF if map is large.
        # But we can assume map is small. 
        # Alternatively, strict SQL way: join with a config df?
        # Or simpler: use a `when/then` chain.
        
        tiny_expr = pl.col("area_rel") < self.config.tiny_object_area # Default
        
        if self.config.tiny_object_area_map:
            # Build expression chain
            # Start with default
            curr_expr = pl.lit(False) 
            # We iterate and build: (class_id == k & area < v) OR ...
            # But specific override should take precedence.
            # actually logic is: threshold = map.get(class_id, default)
            # is_tiny = area < threshold
            
            # To do this efficienty in Polars expressions:
            # We can use `pl.when().then().otherwise()`
            
            # Base 'otherwise' is the global default
            rule = pl.when(pl.lit(False)).then(0.0) # Dummy start
            
            # We construct the threshold column
            thresh_expr = pl.lit(self.config.tiny_object_area)
            
            for cid, thresh in self.config.tiny_object_area_map.items():
                thresh_expr = pl.when(pl.col("class_id") == cid).then(thresh).otherwise(thresh_expr)
            
            q = q.with_columns(thresh_expr.alias("tiny_thresh"))
            tiny_expr = pl.col("area_rel") < pl.col("tiny_thresh")
        else:
             # Just use default
             tiny_expr = pl.col("area_rel") < self.config.tiny_object_area

        q = q.with_columns([
            tiny_expr.alias("is_tiny"),
            (
                (pl.col("x_center") < 0) | (pl.col("x_center") > 1) |
                (pl.col("y_center") < 0) | (pl.col("y_center") > 1)
            ).alias("is_out_of_bounds"),
            
            # Duplicate (exact) - we will add approximate overlap next
            (pl.count("class_id").over(["file_path", "x_center", "y_center", "width", "height"]) > 1).alias("is_duplicate_exact"),
        ])
        
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
        # We need to iterate by (file_path, class_id) groups that have count > 1
        # Polars approach: 
        # 1. Filter candidates
        # 2. Self-join or cross-join within groups?
        # 3. Calculate IoU
        
        # Using a custom apply might be faster for now given the complexity of IoU in pure expressions
        # or we implement vectorized IoU.
        
        # Let's verify size. usage of python loop on groups might be slow if many objects.
        # But usually duplicate labels are rare.
        
        # Strategy:
        # Sort by file, class
        # Iterate groups, if len > 1, compute IoU matrix.
        
        # Optimization: Only check groups with > 1 item
        
        # We will initialize the column as False
        is_high_overlap = np.zeros(len(df), dtype=bool)
        
        # Convert necessary columns to numpy for speed
        # We need a stable index to map back
        df = df.with_row_index("idx")
        
        # Group candidates: count > 1
        candidates = df.filter(pl.count("idx").over(["file_path", "class_id"]) > 1)
        
        if candidates.is_empty():
             return df.with_columns(pl.lit(False).alias("is_high_overlap")).drop("idx")
             
        # Iterate over groups
        # This part iterates in python, can be optimized later if slow
        grouped = candidates.group_by(["file_path", "class_id"])
        
        for _, group in grouped:
             # group is a DataFrame
             if len(group) < 2: continue
             
             boxes = group.select(["x_min", "y_min", "x_max", "y_max"]).to_numpy()
             indices = group["idx"].to_numpy()
             
             # Vectorized IoU n x n
             # Expand
             b1 = boxes[:, None, :] # (N, 1, 4)
             b2 = boxes[None, :, :] # (1, N, 4)
             
             # Intersection
             inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
             inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
             inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
             inter_y2 = np.minimum(b1[..., 3], b2[..., 3])
             
             inter_w = np.maximum(0, inter_x2 - inter_x1)
             inter_h = np.maximum(0, inter_y2 - inter_y1)
             inter_area = inter_w * inter_h
             
             # Union
             area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
             area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
             union_area = area1 + area2 - inter_area
             
             iou = inter_area / (union_area + 1e-6)
             
             # Mask self-comparison
             np.fill_diagonal(iou, 0)
             
             # Check threshold
             # Any box that has an IoU > thresh with ANY other box is a duplicate candidate
             # Note: This marks BOTH as duplicates. User said "mark them".
             # Usually we want to keep one. But for "checking", marking both is better (human review).
             
             has_overlap = np.any(iou > self.config.iou_duplicate_threshold, axis=1)
             
             if np.any(has_overlap):
                 bad_indices = indices[has_overlap]
                 is_high_overlap[bad_indices] = True
                 
        # Update original df
        return df.with_columns(
             pl.Series(name="is_high_overlap", values=is_high_overlap)
        ).drop("idx")
