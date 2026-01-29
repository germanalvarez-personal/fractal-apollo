
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_analyzer import YoloRustAnalyzer

def run_benchmark():
    dataset_path = Path("data_test/labels")
    if not dataset_path.exists():
        print("Data not found. Run generate_data.py first.")
        return

    print("Initializing Analyzer...")
    analyzer = YoloRustAnalyzer(dataset_path)
    
    print("Step 1: Loading Data (Scan)...")
    t0 = time.time()
    # It returns a LazyFrame, so this should be nearly instant
    analyzer.load_data()
    t1 = time.time()
    print(f"Load Data (Scan) Time: {t1 - t0:.4f}s")
    
    print("Step 2: Computing Metrics (Lazy Collect)...")
    t0 = time.time()
    df = analyzer.compute_metrics()
    t1 = time.time()
    print(f"Compute Metrics Time: {t1 - t0:.4f}s")
    print(f"Total Rows Processed: {df.height}")
    
    print("Step 3: Plotting Dashboard...")
    t0 = time.time()
    analyzer.plot_dashboard("benchmarks/dashboard.png")
    t1 = time.time()
    print(f"Plotting Time: {t1 - t0:.4f}s")
    
    print("Step 4: Checking Image I/O (Kornia-rs)...")
    # Point to the images dir implicitly by checking relative paths in analyzer
    # The dummy generator made images in data_test/images
    # The labels are in data_test/labels
    # The analyzer logic tries to find ../images if needed
    analyzer.check_image_io()

if __name__ == "__main__":
    run_benchmark()
