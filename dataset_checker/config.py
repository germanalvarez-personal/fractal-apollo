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
    tiny_object_area: Optional[float] = 0.000125 # If set, strictly overrides stats. If None, uses IQR stats.
    tiny_object_area_map: Optional[dict] = None # Class-specific overrides
    
    # New thresholds for detailed metrics
    duplicate_iou_threshold: float = 0.95 # IoU threshold for approximate duplicates
    truncation_margin: float = 0.01      # Margin from edge to consider object truncated
    
    # Statistical Outliers
    aspect_ratio_z_threshold: float = 3.0
    z_score_sample_min: int = 15      # Min samples required to use Z-score for aspect ratio (falback to abs bounds)
    aspect_ratio_abs_max: float = 5.0 # Absolute max aspect ratio if samples < z_score_sample_min
    aspect_ratio_abs_min: float = 0.2 # Absolute min aspect ratio if samples < z_score_sample_min
    
    area_iqr_low: float = 1.5   # Multiplier for lower bound (Q1 - k*IQR)
    area_iqr_high: float = 2.0  # Multiplier for upper bound (Q3 + k*IQR)
    
    # Duplicates
    iou_duplicate_threshold: float = 0.9
    
    # Oversized
    oversized_safety_floor: float = 0.55 # Absolute max area (80% of image)
    
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
