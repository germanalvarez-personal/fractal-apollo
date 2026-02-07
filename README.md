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
- **Dashboard**: Comprehensive visualization of dataset statistics including frequency, spatial distribution, and integrity metrics.
- **Outlier Reporting**: JSON reports identifying potential dataset anomalies (tiny objects, class-relative outliers, duplicates).
- **Crop Generation**: High-performance tool (`generate_cropset.py`) to create fixed-size crops from large images.
  - Centered object crops with label recalculation.
  - **Background Crops**: Automated generation of empty/background crops using a centroid-recycling strategy to maintain realistic distributions.
  - Multi-threaded processing.

## Dashboard Visualization

The `plot_dashboard()` function generates a 6-panel composite image offering a holistic view of dataset health:

1.  **Top 20 Class Frequency (Top-Left)**
    -   **Type**: Bar Chart.
    -   **Description**: Displays the count of instances for the 20 most frequent classes.
    -   **Purpose**: Identify class imbalance and dominant categories.

2.  **Global Heatmap (Top-Center)**
    -   **Type**: 2D Histogram / Heatmap.
    -   **Description**: Aggregates the center points of all objects into a 100x100 grid.
        -   **X-Axis**: 0 (Left) to 100 (Right).
        -   **Y-Axis**: 0 (Top) to 100 (Bottom) - Standard image coordinates.
    -   **Purpose**: Reveal spatial biases (e.g., objects only appearing in the center) or blind spots in the dataset.

3.  **Area Distribution (Top-Right)**
    -   **Type**: Log-Scaled Box Plot with Overlayed Strip Plot.
    -   **Description**:
        -   Shows the distribution of *Relative Area* (object area / image area) for the top 10 classes.
        -   **Strip Plot**: Individual points are overlayed to show density and reveal specific outliers (colored red).
        -   **Reference Lines**:
            -   **Green Zone**: Optimal area range (configurable, default 1%-20%).
            -   **Red Lines**: Tiny object floor (0.5%) and Oversized ceiling (80%).
    -   **Purpose**: Assess object scale variance and identify if objects are too small/large for the detector.

4.  **Shape Analysis (Bottom-Left)**
    -   **Type**: Hexbin Plot (Log-Log Scale).
    -   **Description**: Plots *Relative Area* vs. *Aspect Ratio*.
        -   **X-Axis**: Relative Area (Object Size).
        -   **Y-Axis**: Aspect Ratio (Width / Height).
    -   **Purpose**: identify clusters of object shapes (e.g., tall/thin vs. wide/short) and their correlation with size.

5.  **Edge Bias (Bottom-Center)**
    -   **Type**: Histogram + KDE.
    -   **Description**: Distribution of the "Edge Proximity" metric (distance to nearest image border).
        -   **0.0**: Object is touching or very close to the edge.
        -   **0.5**: Object is at the exact center of the image.
    -   **Purpose**: Detect if annotations are biased away from edges or if objects are frequently truncated.

6.  **Data Integrity (Bottom-Right)**
    -   **Type**: Summary Table.
    -   **Description**: Quantitative report of dataset flags.
    -   **Metrics**:
        -   **Background Images**: % of images with 0 labels.
        -   **is_tiny**: % of objects below the tiny threshold.
        -   **is_oversized**: % of objects exceeding safety ceiling.
        -   **is_stretched**: % of objects with extreme aspect ratios (Z-score > 3).
        -   **is_duplicate**: % of overlapping objects with IoU > 0.9.
        -   **is_truncated**: % of objects touching borders that are statistically smaller than average.


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

### Crop Generation Tool

To generate a cropped dataset (e.g., for classifier training or detail validation) from a detection dataset:

```bash
uv run generate_cropset.py \
  --images /path/to/images \
  --labels /path/to/labels \
  --output /path/to/output \
  --background-ratio 0.15 \
  --workers 12
```

**Key Arguments:**
- `--background-ratio`: Target percentage (0.0-1.0) of background crops to include (default: 0.15). Background crops are generated only on empty images using object centroid distributions.
- `--workers`: Number of threads to use (defaults to CPU count).


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
