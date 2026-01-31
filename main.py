from yolo_analyzer import YoloRustAnalyzer
import os
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YoloRustAnalyzer: High-performance EDA for YOLO datasets.")
    parser.add_argument("dataset_path", nargs="?", default=".", help="Path to the YOLO dataset (directory containing labels or root).")
    parser.add_argument("--download-coco", action="store_true", help="Download and setup COCO 2017 validation dataset.")
    args = parser.parse_args()

    abs_path = None
    output_dir = "outputs"

    # Logic handling
    if args.download_coco:
        print("Downloading COCO 2017 Validation dataset...")
        # Initialize with dummy, setup_coco will override path
        analyzer = YoloRustAnalyzer("inputs/coco_val2017", output_dir=output_dir) 
        analyzer.setup_coco()
        # Update path for reporting
        abs_path = analyzer.config.dataset_path
        print(f"COCO Ready at {abs_path}")
        
    else:
        # Determine dataset path logic
        if args.dataset_path == ".":
            # Default behavior: look in inputs/
            inputs_dir = Path("inputs")
            if inputs_dir.exists():
                # Take the first subdirectory found? Or just try inputs/?
                subdirs = [x for x in inputs_dir.iterdir() if x.is_dir()]
                if subdirs:
                    print(f"No specific dataset provided, defaulting to found input: {subdirs[0]}")
                    abs_path = subdirs[0].resolve()
                else:
                    # Maybe the inputs folder IS the dataset?
                    abs_path = inputs_dir.resolve()
            else:
                 # Standard current directory fallback if inputs doesn't exist?
                 # Or verify if current dir has dataset?
                 abs_path = Path(".").resolve()
        else:
             abs_path = Path(args.dataset_path).resolve()
        
        if not abs_path or not abs_path.exists():
             print(f"Error: Dataset path '{abs_path}' not found.")
             sys.exit(1)

        analyzer = YoloRustAnalyzer(str(abs_path), output_dir=output_dir)

    print(f"Running analysis on {abs_path}...")
    
    try:
        # Run full pipeline
        df = analyzer.analyze()
        
        # Generate outputs
        analyzer.plot_dashboard()
        analyzer.generate_stratified_mosaic()
        analyzer.generate_outlier_report()
        # analyzer.generate_class_mosaic() # Optional
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        # Print full trace for debugging if needed
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
