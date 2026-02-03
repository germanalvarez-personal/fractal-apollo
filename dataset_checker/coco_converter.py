import json
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

class CocoConverter:
    def __init__(self):
        pass

    def convert(self, json_path: str, output_dir: str, image_dir: str = None, exclude_categories: list = None):
        """
        Converts COCO JSON annotations to YOLO format.
        
        Args:
            json_path: Path to the COCO annotations JSON file.
            output_dir: Directory where YOLO dataset will be saved.
            image_dir: Optional directory containing source images. If provided, images will be copied.
            exclude_categories: List of category names to exclude (e.g. ['transit']).
        """
        json_path = Path(json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        images_output_dir = output_dir / "images"
        images_output_dir.mkdir(exist_ok=True)

        print(f"Loading COCO annotations from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 1. Parse Categories & Filter
        exclude_categories = set(exclude_categories or [])
        categories = {}
        for cat in data['categories']:
            if cat['name'] not in exclude_categories:
                categories[cat['id']] = cat['name']
            else:
                print(f"Excluding category: {cat['name']} (ID: {cat['id']})")

        # Map Sorted IDs to 0..N
        sorted_ids = sorted(categories.keys())
        id_map = {coco_id: i for i, coco_id in enumerate(sorted_ids)}
        yolo_names = [categories[coco_id] for coco_id in sorted_ids] # List of names in order 0..N
        
        print(f"Found {len(categories)} active categories: {yolo_names}")

        # 2. Parse Images
        images = {img['id']: img for img in data['images']}
        
        # 3. Process Annotations
        print("Processing annotations...")
        img_anns = {}
        for ann in tqdm(data['annotations'], desc="Grouping annotations"):
            img_id = ann['image_id']
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        # 4. Generate YOLO Files
        for img_id, anns in tqdm(img_anns.items(), desc="Generating labels"):
            img_info = images.get(img_id)
            if not img_info:
                continue
                
            img_w = img_info['width']
            img_h = img_info['height']
            file_name = img_info['file_name']
            
            # Create label file
            label_name = Path(file_name).stem + ".txt"
            label_path = labels_dir / label_name
            
            lines = []
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in id_map:
                    continue # Skip excluded categories or unmapped IDs
                    
                yolo_cls = id_map[cat_id]
                bbox = ann['bbox'] # [x_min, y_min, w, h]
                
                # Normalize
                x_min, y_min, w, h = bbox
                
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Clip to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            if lines:
                with open(label_path, 'w') as f:
                    f.write("\n".join(lines))
            
            # Copy Image if requested
            if image_dir:
                src_img = Path(image_dir) / file_name
                if src_img.exists():
                    dst_img = images_output_dir / file_name
                    shutil.copy2(src_img, dst_img)

        # 5. Generate data.yaml
        print("Generating data.yaml...")
        self._generate_yaml(output_dir, yolo_names)
        
        print(f"Conversion complete. Output saved to {output_dir}")

    def _generate_yaml(self, output_dir: Path, names_list: list):
        """
        Generates dataset.yaml for YOLO.
        """
        yaml_content = {
            'train': '../train/images',
            'val': '../val/images',
            'test': '../test/images',
            'nc': len(names_list),
            'names': names_list
        }
        
        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
