# YoloRustAnalyzer

A high-performance EDA tool for YOLO datasets, powered by Polars, Rust, and Kornia.

## Features
- **Speed**: Processes 100k+ files in seconds using modular architecture.
- **Lazy Evaluation**: Efficient memory usage with Polars lazy expressions.
- **Parallel Processing**: Multi-threaded image I/O with `kornia-rs`.
- **Modular Design**:
  - `DataLoader`: Robust file discovery and IO validation.
  - `MetricsEngine`: Pure logic for geometric calculations.
  - `Visualizer`: Parallelized mosaic generation and dashboard plotting.
- **Dashboard**: Comprehensive visualization of dataset statistics including:
  - Class frequency.
  - Global heatmaps.
  - **Log-Scaled Box Plots** for object area distribution.
  - Data integrity flags.
  - **Outlier Reporting**: JSON reports identifying potential dataset anomalies (tiny objects, class-relative outliers, duplicates).

## Usage

### Command Line
Run the tool directly from the command line using `uv`:

```bash
# Explicit yaml path
uv run main.py /path/to/dataset /path/to/dataset.yaml

# Implicit (auto-detect yaml in dataset root)
uv run main.py /path/to/dataset
```
Or simply run it in the current directory (if paths are valid):
```bash
uv run main.py . ./dataset.yaml
```

### Python API

```python
from yolo_analyzer import YoloRustAnalyzer

# Initialize analyzer
analyzer = YoloRustAnalyzer("path/to/dataset")

# Run analysis (loads data, computes metrics, validates)
df = analyzer.analyze()

# Generate visual outputs
analyzer.plot_dashboard(save_path="dashboard.png")
analyzer.generate_stratified_mosaic(save_path="mosaic.png")
analyzer.generate_outlier_report(filename="outliers.json")
```

## Configuration

Configuration is managed in `dataset_checker/config.py`. You can modify the `DatasetConfig` class to tune the analysis.

| Category | Option | Default | Description |
| :--- | :--- | :--- | :--- |
| **Files** | `img_ext` | `jpg` | Image file extension to look for. |
| | `label_ext` | `txt` | Label file extension (YOLO format). |
| **Inliers** | `optimal_area_min` | `0.01` | Minimum relative area (1%) for "Optimal Zone" visualization. |
| | `optimal_area_max` | `0.20` | Maximum relative area (20%) for "Optimal Zone" visualization. |
| **Outliers** | `tiny_object_area` | `0.005` | Absolute floor (0.5%) below which objects are flagged as `is_tiny`. |
| | `oversized_safety_floor` | `0.80` | Absolute ceiling (80%) above which objects are flagged as `is_oversized`. |
| | `area_iqr_low` | `1.5` | IQR multiplier for lower bound (statistically small). |
| | `area_iqr_high` | `2.0` | IQR multiplier for upper bound (statistically large). |
| | `aspect_ratio_z_threshold`| `3.0` | Z-score threshold for identifying `is_stretched` objects. |
| **Quality** | `iou_duplicate_threshold`| `0.9` | IoU threshold above which overlapping objects are flagged as `is_duplicate`. |
| **Vis** | `mosaic_tile_size` | `128` | Pixel size (NxN) for each tile in the stratified mosaic. |
