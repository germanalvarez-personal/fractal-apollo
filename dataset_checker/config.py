from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DatasetConfig:
    dataset_path: Path
    img_ext: str = 'jpg'
    label_ext: str = 'txt'
    images_path_override: Optional[Path] = None
    
    # Thresholds
    tiny_object_area: Optional[float] = None # If set, strictly overrides stats. If None, uses IQR stats.
    tiny_object_area_map: Optional[dict] = None # Class-specific overrides
    
    # New thresholds for detailed metrics
    duplicate_iou_threshold: float = 0.95 # IoU threshold for approximate duplicates
    truncation_margin: float = 0.0        # Margin from edge to consider object truncated (0.0 = exact touch)
    truncation_quantile: float = 0.05     # Quantile threshold for size check (0.1 = 10th percentile).
    
    # Statistical Outliers
    aspect_ratio_z_threshold: float = 4.0
    z_score_sample_min: int = 15      # Min samples required to use Z-score for aspect ratio (falback to abs bounds)
    aspect_ratio_abs_max: float = 5.0 # Absolute max aspect ratio if samples < z_score_sample_min
    aspect_ratio_abs_min: float = 0.2 # Absolute min aspect ratio if samples < z_score_sample_min
    
    area_iqr_low: float = 0.1   # Multiplier for lower bound (Q1 - k*IQR)
    area_iqr_low_map: Optional[dict] = None # Class-specific IQR multiplier overrides for lower bound
    area_iqr_high: float = 5.0  # Multiplier for upper bound (Q3 + k*IQR)
    area_iqr_high_map: Optional[dict] = None # Class-specific IQR multiplier overrides for upper bound
    
    # Duplicates
    iou_duplicate_threshold: float = 0.9
    
    # Oversized
    oversized_safety_floor: float = None # Absolute max area (55% of image)
    oversized_safety_floor_map: Optional[dict] = None # Class-specific absolute max area overrides
    
    # Visualization Ranges
    optimal_area_min: float = 0.01
    optimal_area_max: float = 0.20
    
    # Visualization defaults
    mosaic_tile_size: int = 128
    mosaic_grid_w: int = 10
    mosaic_grid_h: int = 8
    mosaic_min_samples: int = 4   # Minimum samples per class to generate a stratified mosaic strip
    mosaic_max_classes: int = 40  # Maximum number of classes to include in the stratified mosaic
    
    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        if self.images_path_override:
            self.images_path_override = Path(self.images_path_override)
        if self.tiny_object_area_map is None:
            self.tiny_object_area_map = {}
        if self.area_iqr_low_map is None:
            self.area_iqr_low_map = {
            0: 0.214,
            1: 0.199,
            2: 0.191,
            3: 0.087,
            4: 0.541,
        }
        if self.area_iqr_high_map is None:
            self.area_iqr_high_map = {
            0: 1.5,
            1: 1.5,
            2: 1.5,
            3: 1.5,
            4: 1.5,
        }
        if self.oversized_safety_floor_map is None:
            self.oversized_safety_floor_map = {}
