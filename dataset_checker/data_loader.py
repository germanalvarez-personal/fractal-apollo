import polars as pl
from pathlib import Path
import glob
import kornia_rs
import fiftyone as fo
import fiftyone.zoo as foz
import yaml
from .config import DatasetConfig
import concurrent.futures

class DataLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.class_names = {}
        self.total_images = 0
        
    def _resolve_image_path(self, label_path: Path) -> Path:
        """
        Robustly resolves image path from label path.
        Strategy:
        1. If override provided, use it.
        2. Look for 'labels' in path parents and swap with 'images'.
        3. Fallback: sibling directory.
        """
        if self.config.images_path_override:
            # Assuming parallel structure or flat structure in override
            # Try to match relative path from dataset root? 
            # Or just name matching? Let's try name matching for simplicity first 
            # unless structure is known.
            return self.config.images_path_override / label_path.with_suffix(f".{self.config.img_ext}").name

        # Standard YOLO structure: .../labels/train/file.txt -> .../images/train/file.jpg
        # We find the right-most 'labels' part to replace, or check parents.
        parts = list(label_path.parts)
        
        # We iterate backwards to find the deep-most 'labels' folder (closest to file)
        # to handle nested 'labels/labels' cases correctly? 
        # Actually usually it is `dataset/labels/split/file.txt`
        
        try:
            # Find the index of 'labels' - searching from right to left is safer for nested?
            # actually standard python index finds first from left.
            # Let's try to be smart.
            
            # Use pathlib replacement if applicable
            # Check if 'labels' is in the path
            if "labels" in parts:
                # Replace the last occurrence of labels? or valid parent?
                # Let's stick to the heuristic: replace specific parent "labels" with "images"
                # constructing a new path.
                
                # Check for standard YOLO parent structure
                # parent is dir containing file. parent.parent might be 'labels' or 'val'.
                
                # Simple swap:
                new_parts = list(parts)
                # Find all indices of 'labels'
                indices = [i for i, x in enumerate(parts) if x == "labels"]
                if indices:
                    # Replace the relevant one. Usually the one that is a parent.
                    # Let's replace the last one found, assuming it's the split container
                    idx = indices[-1]
                    new_parts[idx] = "images"
                    return Path(*new_parts).with_suffix(f".{self.config.img_ext}")
            
            # Fallback: just change extension
            return label_path.with_suffix(f".{self.config.img_ext}")
            
        except Exception:
            return label_path.with_suffix(f".{self.config.img_ext}")

    def load_data(self) -> pl.LazyFrame:
        """
        High-speed parsing of YOLO .txt files using Polars scan_csv.
        """
        pattern = str(self.config.dataset_path / f"**/*.{self.config.label_ext}")
        
        schema = {
            "class_id": pl.Int32,
            "x_center": pl.Float32,
            "y_center": pl.Float32,
            "width": pl.Float32,
            "height": pl.Float32
        }

        try:
            q = pl.scan_csv(
                pattern,
                separator=' ',
                has_header=False,
                new_columns=["class_id", "x_center", "y_center", "width", "height"],
                schema=schema,
                include_file_paths="file_path"
            )
            
            # Estimate total images
            self._count_total_images()
            
            return q
        except Exception as e:
            print(f"Error scanning files: {e}")
            raise

    def _count_total_images(self):
        # Heuristic count
        # Search relative to dataset_path
        # If dataset_path ends in 'labels', try sibling 'images'
        search_path = self.config.dataset_path
        if self.config.dataset_path.name == "labels":
            search_path = self.config.dataset_path.parent / "images"
        elif "labels" in self.config.dataset_path.parts:
             # Try to switch
             search_path = self._resolve_image_path(self.config.dataset_path) # Might be file path logic though
             if not search_path.exists() and self.config.dataset_path.parent.name == "labels":
                 search_path = self.config.dataset_path.parent.parent / "images"
        
        # If override is set
        if self.config.images_path_override:
            search_path = self.config.images_path_override

        try:
            if search_path.exists():
                self.total_images = len(glob.glob(str(search_path / f"**/*.{self.config.img_ext}"), recursive=True))
            else:
                self.total_images = 0
        except:
            self.total_images = 0

    def load_class_names(self, yaml_path: Path):
        """Loads class names from a YOLO dataset.yaml file."""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    if isinstance(names, dict):
                         self.class_names = {int(k): v for k, v in names.items()}
                    elif isinstance(names, list):
                         self.class_names = {i: v for i, v in enumerate(names)}
            print(f"Loaded {len(self.class_names)} class names.")
        except Exception as e:
            print(f"Failed to load class names: {e}")

    def setup_coco(self, download_dir="coco_val2017"):
        """Automated COCO Val2017 Setup."""
        download_path = Path(download_dir)
        labels_path = download_path / "labels"
        if labels_path.exists() and len(list(labels_path.glob("*.txt"))) > 0:
            print(f"COCO dataset found at {download_path}")
            self.config.dataset_path = labels_path
            return

        print("COCO dataset not found. Downloading via FiftyOne...")
        dataset = foz.load_zoo_dataset("coco-2017", split="validation")
        
        print("Exporting to YOLO format...")
        export_dir = download_path / "yolo_export"
        dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
        )
        
        self.config.dataset_path = export_dir / "labels" / "val"
        print(f"COCO setup complete. Dataset path: {self.config.dataset_path}")
        
        yaml_path = export_dir / "dataset.yaml"
        if yaml_path.exists():
            self.load_class_names(yaml_path)

    def validate_dataset(self, df: pl.DataFrame, sample_frac=0.05):
        """
        Consolidated check for Ghost Labels and Image IO using ThreadPool.
        """
        print(f"Running Dataset Validation on {sample_frac*100}% sample...")
        
        unique_paths = df["file_path"].unique().to_list()
        import random
        sample_size = int(len(unique_paths) * sample_frac)
        sampled_paths = random.sample(unique_paths, sample_size) if sample_size < len(unique_paths) else unique_paths
        
        ghosts = 0
        read_errors = 0
        
        # Check for kornia support
        try:
             k_read = kornia_rs.read_image_jpeg
        except:
             print("Kornia-rs missing, using PIL for validation fallback (slower).")
             from PIL import Image
             k_read = None

        def check_file(p_str):
            p = Path(p_str)
            img_p = self._resolve_image_path(p)
            
            if not img_p.exists():
                return 1, 0 # ghost, no_error
            
            try:
                if k_read:
                    _ = k_read(str(img_p), "rgb")
                else:
                    with Image.open(img_p) as img:
                        img.verify()
                return 0, 0
            except Exception:
                return 0, 1 # no_ghost, error
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(check_file, sampled_paths))
            
        for g, e in results:
            ghosts += g
            read_errors += e
            
        print(f"Validation Report (Sample: {len(sampled_paths)}):")
        print(f"  Missing Images: {ghosts}")
        print(f"  Corrupt/Unreadable: {read_errors}")
        if len(sampled_paths) > 0:
             print(f"  Ghost Rate: {(ghosts/len(sampled_paths))*100:.2f}%")
