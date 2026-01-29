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

## Usage

### Command Line
Run the tool directly from the command line using `uv`:

```bash
uv run main.py /path/to/dataset
```
Or simply run it in the current directory:
```bash
uv run main.py .
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
```

## Configuration
Configuration is handled in `dataset_checker/config.py`. You can adjust:
- Mosaic tile/grid sizes.
- Quality thresholds (tiny objects, IoU).
- File extensions.
