
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_analyzer import YoloRustAnalyzer

def run_coco_benchmark():
    print("=== Starting COCO Val2017 Benchmark ===")
    
    # Initialize Analyzer pointing to where we expect COCO to be
    # We will use 'coco_data' as download dir
    coco_dir = Path("coco_data")
    analyzer = YoloRustAnalyzer("placeholder") 
    
    # 1. Pipeline Setup (Auto-download & Convert)
    print("Step 1: Metric - Automated COCO Setup")
    analyzer.setup_coco(download_dir=str(coco_dir))
    
    if not analyzer.dataset_path.exists():
        print("COCO setup failed.")
        return

    # Load Data
    analyzer.load_data()
    print("Loading complete.")
    
    # Compute Metrics
    print("Step 2: Metric - High Speed Processing")
    analyzer.compute_metrics()
    
    df = analyzer.df
    
    # 3. Integrity Stress Test: Verify "Tiny" objects (Area < 0.005)
    print("Step 3: Metric - Integrity Stress Test (Tiny Objects)")
    tiny_count = df["is_tiny"].sum()
    tiny_pct = (tiny_count / df.height) * 100
    print(f"Tiny Objects Count: {tiny_count}")
    print(f"Tiny Objects %: {tiny_pct:.2f}%")
    
    if 6.0 <= tiny_pct <= 10.0:
        print("✅ PASS: Tiny object percentage within expected range (6-10%).")
    else:
        print(f"⚠️ WARNING: Tiny object percentage {tiny_pct:.2f}% outside expected range (6-10%).")

    # 4. Diagnostic: Small Object Heatmap
    print("\nStep 4: Metric - Spatial Bias Check (Small Object Heatmap)")
    # Generate Heatmap for Small Objects (Area < 0.01)
    heatmap_small = analyzer.generate_heatmap(bins=100, area_max=0.01)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_small, cmap="inferno", cbar=True)
    plt.title("COCO Val2017 - Small Object Density (Area < 0.01)")
    plt.gca().invert_yaxis()
    plt.savefig("coco_small_heatmap.png")
    print("Saved Small Object Heatmap to coco_small_heatmap.png")

    # 5. Diagnostic: Ghost Label Check
    print("\nStep 5: Metric - Ghost Label Check (kornia-rs)")
    # Sampling 5% per requirement
    analyzer.check_ghost_labels(sample_frac=0.05)

    # 6. Generate Full Dashboard
    print("\nStep 6: Generating Full Dashboard...")
    analyzer.plot_dashboard("coco_dashboard.png")
    
    # 7. Class Analysis
    print("\nStep 7: Class-Specific Analysis")
    analyzer.plot_heatmap_grid("coco_heatmap_grid.png")
    # Stratified (2x2) Mosaic
    analyzer.generate_stratified_mosaic("coco_stratified_mosaic.png")
    
    print("\n=== COCO Benchmark Complete ===")

if __name__ == "__main__":
    run_coco_benchmark()
