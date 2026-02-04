from yolo_analyzer import YoloRustAnalyzer
import os
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YoloRustAnalyzer: High-performance EDA for YOLO datasets.")
    parser.add_argument("dataset_path", nargs="?", help="Path to the YOLO dataset (directory containing labels or root).")
    parser.add_argument("dataset_yaml", nargs="?", help="Optional path to the dataset.yaml file containing class names.")
    parser.add_argument("--download-coco", action="store_true", help="Download and setup COCO 2017 validation dataset.")
    
    # Conversion Args
    parser.add_argument("--convert-coco", help="Path to COCO JSON annotations to convert to YOLO.")
    parser.add_argument("--images-dir", help="Directory containing source images for COCO conversion (optional).")
    parser.add_argument("--output-dir", default="converted_dataset", help="Output directory for converted YOLO dataset.")
    parser.add_argument("--exclude-category", nargs="*", help="List of category names to exclude from conversion (e.g. transit).")
    
    # Analysis Configuration Overrides
    parser.add_argument("--img-ext", default="jpg", help="Image extension to look for (default: jpg).")
    parser.add_argument("--tiny-threshold", type=float, help="Override global tiny object area threshold (e.g. 0.003).")
    
    args = parser.parse_args()
    
    # Mode Selection
    if args.convert_coco:
        from dataset_checker.coco_converter import CocoConverter
        converter = CocoConverter()
        print(f"Converting COCO: {args.convert_coco} -> {args.output_dir}")
        if args.exclude_category:
            print(f"Excluding categories: {args.exclude_category}")
        converter.convert(args.convert_coco, args.output_dir, args.images_dir, exclude_categories=args.exclude_category)
        sys.exit(0)

    # Standard Analysis Mode requires dataset path
    if not args.dataset_path and not args.download_coco:
        parser.print_help()
        sys.exit(1)

    abs_path = None
    yaml_path = None
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
        abs_path = Path(args.dataset_path).resolve()
        
        # Determine YAML path
        if args.dataset_yaml:
            yaml_path = Path(args.dataset_yaml).resolve()
        else:
             # Fallback search
             candidates = [
                 abs_path / "dataset.yaml",
                 abs_path / "data.yaml",
                 abs_path.parent / "dataset.yaml" # In case abs_path is 'labels' or 'val'
             ]
             
             found_yaml = False
             for c in candidates:
                 if c.exists():
                     yaml_path = c
                     print(f"Auto-detected dataset config: {yaml_path}")
                     found_yaml = True
                     break
            
             if not found_yaml and abs_path.exists():
                 # Try limited recursive search (depth 2)
                 print("Deep searching for dataset.yaml...")
                 try:
                     possible = sorted(list(abs_path.rglob("dataset.yaml")) + list(abs_path.rglob("data.yaml")))
                     if possible:
                         # Prefer shortest path or one named dataset.yaml
                         yaml_path = possible[0]
                         print(f"Auto-detected dataset config (deep search): {yaml_path}")
                 except Exception as e:
                     print(f"Deep search failed: {e}")
        
        if not abs_path or not abs_path.exists():
             print(f"Error: Dataset path '{abs_path}' not found.")
             sys.exit(1)
        
        # Build Config Overrides
        config_overrides = {
            "img_ext": args.img_ext
        }
        if args.tiny_threshold is not None:
            config_overrides["tiny_object_area"] = args.tiny_threshold
             
        analyzer = YoloRustAnalyzer(str(abs_path), output_dir=output_dir, **config_overrides)

        if yaml_path and yaml_path.exists():
             analyzer.load_class_names(yaml_path)
        else:
             print("Warning: No dataset.yaml found. Class names will be missing in visualizations.")

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
