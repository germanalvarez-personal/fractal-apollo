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
    tiny_object_area: float = 0.005 # Fallback global threshold
    tiny_object_area_map: Optional[dict] = None # Class-specific overrides
    
    # Statistical Outliers
    aspect_ratio_z_threshold: float = 3.0
    
    # Duplicates
    iou_duplicate_threshold: float = 0.9
    
    # Oversized
    oversized_object_area: float = 0.6
    
    # Visualization Ranges
    optimal_area_min: float = 0.01
    optimal_area_max: float = 0.20
    
    # Visualization defaults
    mosaic_tile_size: int = 128
    mosaic_grid_w: int = 10
    mosaic_grid_h: int = 8
    
    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        if self.images_path_override:
            self.images_path_override = Path(self.images_path_override)
        if self.tiny_object_area_map is None:
            self.tiny_object_area_map = {}
