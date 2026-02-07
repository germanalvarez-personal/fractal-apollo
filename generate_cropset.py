import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import kornia_rs
import shutil

# --- Configuration ---
# Target crop size
CROP_SIZE = 1280
# Minimum visible area ratio to keep a label (avoid keeping slivers of objects)
MIN_VISIBILITY_RATIO = 0.2

def get_valid_crop_window(img_w, img_h, center_x, center_y, crop_size):
    """
    Calculates the top-left (x1, y1) coordinates of a crop centered at (center_x, center_y).
    Ensures the crop strictly stays within image bounds [0, img_w] x [0, img_h].
    """
    half_crop = crop_size // 2
    
    # Desired top-left
    x1 = center_x - half_crop
    y1 = center_y - half_crop
    
    # Clamp x
    if x1 < 0:
        x1 = 0
    elif x1 + crop_size > img_w:
        x1 = img_w - crop_size
        
    # Clamp y
    if y1 < 0:
        y1 = 0
    elif y1 + crop_size > img_h:
        y1 = img_h - crop_size
        
    return int(max(0, x1)), int(max(0, y1))

def load_class_names(yaml_path):
    """
    Parses a YOLO dataset.yaml file to extract class names.
    Supports list format:
      names:
        0: person
        1: car
    Or list format:
      names: ['person', 'car']
    Or simple list:
      names:
        - person
        - car
    """
    names = {}
    try:
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
        
        in_names_block = False
        is_dict_format = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            if stripped.startswith('names:'):
                # Check for inline list
                if '[' in stripped and ']' in stripped:
                    content = stripped.split('[')[1].split(']')[0]
                    parts = [p.strip().strip("'").strip('"') for p in content.split(',')]
                    for i, name in enumerate(parts):
                        names[i] = name
                    break # Done
                
                in_names_block = True
                continue
            
            if in_names_block:
                # Check if we exited the block (new key)
                if ':' in stripped and not stripped.startswith('-') and not stripped[0].isdigit():
                    # Could be "nc: 8" or similar
                    if not (stripped[0].isdigit() and ':' in stripped): # Dict key check
                         break
                
                # List item "- name"
                if stripped.startswith('- '):
                    idx = len(names)
                    name = stripped[2:].strip().strip("'").strip('"')
                    names[idx] = name
                # Dict item "0: name"
                elif ':' in stripped:
                    parts = stripped.split(':')
                    if parts[0].strip().isdigit():
                        idx = int(parts[0].strip())
                        name = parts[1].strip().strip("'").strip('"')
                        names[idx] = name
    except Exception as e:
        print(f"Warning: Failed to parse class names from {yaml_path}: {e}")
        
    return names

def process_image_job(image_path, label_path, output_images_dir, output_labels_dir, debug_dir, debug_limit_counter, class_names):
    """
    Worker function to process a single image.
    1. Reads labels.
    2. Reads image (lazily or fully if needed).
    3. Iterates over objects -> Generates unique crops.
    4. Saves crops and new labels.
    """
    try:
        # 1. Parse Labels First (Fastest fail)
        if not label_path.exists():
            return []
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return []

        # Parse valid lines: class x_c y_c w h
        # Store as list of dicts or numpy array for speed. List is fine for typical object counts.
        objects = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                objects.append({
                    'class_id': int(parts[0]),
                    'x_c': float(parts[1]),
                    'y_c': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4]),
                    'raw_line': line
                })
        
        if not objects:
            return []

        # 2. Read Image High-Speed (Kornia-RS)
        try:
            img_tensor = kornia_rs.read_image_jpeg(str(image_path), "rgb")
            img = np.array(img_tensor) # Ensure numpy
        except:
             # Fallback to cv2 if kornia fails (e.g. png not jpeg)
             img = cv2.imread(str(image_path))
             if img is None:
                 return [f"Failed to read image: {image_path}"]
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]
        
        # 3. Generate Crops
        generated_files = []
        
        for idx, target_obj in enumerate(objects):
            # Calculate absolute center of target
            abs_cx = target_obj['x_c'] * img_w
            abs_cy = target_obj['y_c'] * img_h
            
            # Calculate Crop Window
            crop_x1, crop_y1 = get_valid_crop_window(img_w, img_h, abs_cx, abs_cy, CROP_SIZE)
            crop_x2 = crop_x1 + CROP_SIZE
            crop_y2 = crop_y1 + CROP_SIZE
            
            # Perform Crop (NumPy slicing is zero-copy usually, but we need to save it)
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 4. Find all objects inside this crop and adjust coordinates
            new_labels = []
            
            for obj in objects:
                # Convert obj to absolute pixels
                ox_c = obj['x_c'] * img_w
                oy_c = obj['y_c'] * img_h
                ow = obj['w'] * img_w
                oh = obj['h'] * img_h
                
                ox1 = ox_c - ow / 2
                oy1 = oy_c - oh / 2
                ox2 = ox_c + ow / 2
                oy2 = oy_c + oh / 2
                
                # Intersection with Crop Window
                # Crop is (crop_x1, crop_y1, crop_x2, crop_y2)
                
                ix1 = max(crop_x1, ox1)
                iy1 = max(crop_y1, oy1)
                ix2 = min(crop_x2, ox2)
                iy2 = min(crop_y2, oy2)
                
                if ix2 > ix1 and iy2 > iy1:
                    # Visible area exists
                    inter_w = ix2 - ix1
                    inter_h = iy2 - iy1
                    inter_area = inter_w * inter_h
                    original_area = ow * oh
                    
                    if original_area > 0 and (inter_area / original_area) >= MIN_VISIBILITY_RATIO:
                        # Valid object to keep.
                        # Calculate NEW bounding box relative to crop
                        
                        # New relative coordinates (0 to CROP_SIZE)
                        new_x1 = ix1 - crop_x1
                        new_y1 = iy1 - crop_y1
                        new_x2 = ix2 - crop_x1
                        new_y2 = iy2 - crop_y1
                        
                        # Convert back to YOLO format (center normalized 0-1 relative to CROP_SIZE)
                        new_w_abs = new_x2 - new_x1
                        new_h_abs = new_y2 - new_y1
                        new_cx_abs = new_x1 + new_w_abs / 2
                        new_cy_abs = new_y1 + new_h_abs / 2
                        
                        norm_x = new_cx_abs / CROP_SIZE
                        norm_y = new_cy_abs / CROP_SIZE
                        norm_w = new_w_abs / CROP_SIZE
                        norm_h = new_h_abs / CROP_SIZE
                        
                        # Clamp 0-1
                        norm_x = min(max(norm_x, 0.0), 1.0)
                        norm_y = min(max(norm_y, 0.0), 1.0)
                        norm_w = min(max(norm_w, 0.0), 1.0)
                        norm_h = min(max(norm_h, 0.0), 1.0)
                        
                        new_labels.append(f"{obj['class_id']} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            # Save Output
            if new_labels:
                base_name = image_path.stem
                # Naming convention: {original_name}_obj_{idx}
                out_name = f"{base_name}_obj_{idx}"
                
                out_img_path = output_images_dir / f"{out_name}.jpg"
                out_lbl_path = output_labels_dir / f"{out_name}.txt"
                
                # Write Image (Use cv2 for default reliable writing)
                crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_img_path), crop_bgr)
                
                with open(out_lbl_path, 'w') as f_out:
                    f_out.write('\n'.join(new_labels))
                
                generated_files.append(out_name)
                
                # Debug Visualization (Thread-safe counter check)
                if len(debug_limit_counter) < 10:
                    debug_limit_counter.append(1) # Atomic-ish enough for this purpose
                    
                    debug_img = crop_bgr.copy()
                    h_crop, w_crop = debug_img.shape[:2]
                    
                    for line in new_labels:
                        parts = line.split()
                        cls_id = int(parts[0])
                        nx, ny, nw, nh = map(float, parts[1:])
                        
                        dx = int(nx * w_crop)
                        dy = int(ny * h_crop)
                        dw = int(nw * w_crop)
                        dh = int(nh * h_crop)
                        
                        l = int(dx - dw/2)
                        t = int(dy - dh/2)
                        r = int(dx + dw/2)
                        b = int(dy + dh/2)
                        
                        cv2.rectangle(debug_img, (l, t), (r, b), (0, 255, 0), 2)
                        
                        label_text = f"{cls_id}"
                        if class_names and cls_id in class_names:
                            label_text = f"{cls_id}: {class_names[cls_id]}"
                            
                        cv2.putText(debug_img, label_text, (l, t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    debug_out_path = debug_dir / f"DEBUG_{out_name}.jpg"
                    cv2.imwrite(str(debug_out_path), debug_img)

        return generated_files

    except Exception as e:
        return [f"Error processing {image_path}: {str(e)}"]

def main():
    parser = argparse.ArgumentParser(description="Generate validation crop dataset from 4K YOLO data.")
    parser.add_argument("--images", type=str, required=True, help="Path to source images directory")
    parser.add_argument("--labels", type=str, required=True, help="Path to source labels directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output root directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker threads")
    parser.add_argument("--background-ratio", type=float, default=0.15, help="Target ratio of background crops (default: 0.15)")
    
    args = parser.parse_args()
    
    src_images_dir = Path(args.images)
    src_labels_dir = Path(args.labels)
    output_root = Path(args.output)
    
    # 1. Look for dataset.yaml for class names
    print("Looking for dataset.yaml...")
    possible_paths = [
        src_images_dir.parent / "dataset.yaml",
        src_images_dir.parent / "data.yaml",
        src_labels_dir.parent / "dataset.yaml",
        src_labels_dir.parent / "data.yaml",
        Path("dataset.yaml").resolve()
    ]
    
    dataset_yaml_path = None
    for p in possible_paths:
        if p.exists():
            dataset_yaml_path = p
            break
            
    class_names = {}
    if dataset_yaml_path:
        print(f"Found dataset configuration: {dataset_yaml_path}")
        class_names = load_class_names(dataset_yaml_path)
        print(f"Loaded {len(class_names)} classes: {class_names}")
    else:
        print("Warning: dataset.yaml or data.yaml not found. Debug visualization will use IDs only.")
    
    # Setup Dirs
    out_images_dir = output_root / "images"
    out_labels_dir = output_root / "labels"
    out_debug_dir = output_root / "debug_visualization"
    
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy dataset.yaml if found
    if dataset_yaml_path:
        try:
            shutil.copy(dataset_yaml_path, output_root / "dataset.yaml")
            print(f"Copied {dataset_yaml_path.name} to output directory.")
        except Exception as e:
            print(f"Warning: Failed to copy {dataset_yaml_path.name}: {e}")
    
    # Gather tasks
    print(f"Scanning {src_images_dir}...")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in src_images_dir.iterdir() 
        if f.suffix.lower() in valid_extensions and f.is_file()
    ]
    
    # Sort for consistency
    image_files.sort()

    # --- Phase 1: Pre-scan for Centroids and Empty Images ---
    print("Pre-scanning dataset for object centroids and empty images...")
    all_centroids = []
    empty_images = []
    total_objects = 0
    
    # Fast scan of labels
    # We can do this in parallel or serial. Serial is likely fast enough for FS check unless huge.
    # Let's do it in parallel to be safe.
    
    def scan_label_file(img_path):
        lbl_name = img_path.stem + ".txt"
        lbl_path = src_labels_dir / lbl_name
        
        centroids = []
        has_labels = False
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    has_labels = True
                    # class x y w h
                    centroids.append((float(parts[1]), float(parts[2])))
        
        return img_path, centroids, has_labels

    with ThreadPoolExecutor(max_workers=args.workers) as scan_executor:
        future_to_scan = {scan_executor.submit(scan_label_file, img): img for img in image_files}
        
        for future in tqdm(as_completed(future_to_scan), total=len(image_files), desc="Pre-scanning"):
            img_p, cents, has_lbl = future.result()
            if has_lbl:
                all_centroids.extend(cents)
                total_objects += len(cents)
            else:
                empty_images.append(img_p)

    print(f"Total objects found: {total_objects}")
    print(f"Empty images (potential background sources): {len(empty_images)}")
    
    # --- Phase 2: Target Calculation ---
    target_bg_count = 0
    crops_per_empty = 0
    
    if args.background_ratio > 0 and args.background_ratio < 1.0:
        # B = (R * O) / (1 - R)
        if total_objects > 0:
            target_bg_count = (total_objects * args.background_ratio) / (1.0 - args.background_ratio)
            target_bg_count = int(np.ceil(target_bg_count))
            
            if len(empty_images) > 0:
                crops_per_empty = int(np.ceil(target_bg_count / len(empty_images)))
                print(f"Target Background Crops: {target_bg_count} (Ratio: {args.background_ratio})")
                print(f"Distributing ~{crops_per_empty} crops per empty image ({len(empty_images)} images)")
            else:
                print("Warning: Background ratio requested but NO empty images found. Skipping background generation.")
        else:
            print("Warning: No objects found in dataset. Cannot calculate background ratio.")
    
    # --- Phase 3: Processing ---
            
    print(f"Found {len(image_files)} images. Starting processing with {args.workers} workers...")
    print(f"Output: {output_root}")
    print(f"Crop Size: {CROP_SIZE}x{CROP_SIZE}")
    
    # Shared 'counter' for debug limits (list is thread-safe for append in CPython usually, or close enough)
    debug_counter = []
    
    processed_count = 0
    generated_count = 0
    generated_bg_count = 0
    errors = []
    
    # Convert empty_images list to a set for fast lookup in worker
    empty_images_set = set(empty_images)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        # Map: image_file -> future
        future_to_file = {}
        for img_path in image_files:
            # Find matching label
            lbl_name = img_path.stem + ".txt"
            lbl_path = src_labels_dir / lbl_name
            
            # Determine job type
            is_empty = img_path in empty_images_set
            
            # If it's an empty image and we need background crops
            bg_crops_to_gen = crops_per_empty if is_empty and crops_per_empty > 0 else 0
            
            # We need to pass centroids to worker if it's an empty image job
            # To avoid pickling massive list for every job, we might rely on 'fork' behavior or shared memory?
            # Actually, standard ThreadPoolExecutor shares memory. So passing reference is fine.
            # But we should only pass it if needed.
            
            task_centroids = all_centroids if bg_crops_to_gen > 0 else None

            future = executor.submit(
                process_image_job_v2, # Updated worker name
                img_path,
                lbl_path,
                out_images_dir,
                out_labels_dir,
                out_debug_dir,
                debug_counter,
                class_names,
                bg_crops_to_gen,
                task_centroids
            )
            future_to_file[future] = img_path
            
        # Progress bar
        with tqdm(total=len(image_files), desc="Processing") as pbar:
            for future in as_completed(future_to_file):
                img_path = future_to_file[future]
                try:
                    results = future.result()
                    # Results is list of generated files or error strings
                    if results and isinstance(results[0], str) and results[0].startswith("Error"):
                        errors.append(results[0])
                    else:
                        processed_count += 1
                        generated_count += len(results)
                        
                except Exception as exc:
                    errors.append(f"{img_path.name}: {exc}")
                
                pbar.update(1)
                pbar.set_postfix({"Crops": generated_count, "Errors": len(errors)})

    print("\nProcessing Complete.")
    print(f"Processed Images: {processed_count}")
    print(f"Total Generated Crops: {generated_count}")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for e in errors[:10]:
            print(f" - {e}")
        if len(errors) > 10:
            print(f" ... and {len(errors) - 10} more.")

def process_image_job_v2(image_path, label_path, output_images_dir, output_labels_dir, debug_dir, debug_limit_counter, class_names, num_bg_crops=0, all_centroids=None):
    """
    Worker function to process a single image.
    Handles both object-centered crops and background crops on empty images.
    """
    try:
        # 1. Parse Labels (if exist)
        objects = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    objects.append({
                        'class_id': int(parts[0]),
                        'x_c': float(parts[1]),
                        'y_c': float(parts[2]),
                        'w': float(parts[3]),
                        'h': float(parts[4]),
                        'raw_line': line
                    })
        
        # If we expect crops (objects exist OR strictly empty with bg request) 
        if not objects and num_bg_crops == 0:
            return []

        # 2. Read Image
        try:
            img_tensor = kornia_rs.read_image_jpeg(str(image_path), "rgb")
            img = np.array(img_tensor)
        except:
             img = cv2.imread(str(image_path))
             if img is None:
                 return [f"Failed to read image: {image_path}"]
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]
        generated_files = []
        
        # --- A. Object Centered Crops ---
        for idx, target_obj in enumerate(objects):
            # Same logic as before
            abs_cx = target_obj['x_c'] * img_w
            abs_cy = target_obj['y_c'] * img_h
            
            crop_x1, crop_y1 = get_valid_crop_window(img_w, img_h, abs_cx, abs_cy, CROP_SIZE)
            crop_x2 = crop_x1 + CROP_SIZE
            crop_y2 = crop_y1 + CROP_SIZE
            
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Recalculate labels
            new_labels = recalculate_labels(objects, crop_x1, crop_y1, img_w, img_h)
            
            if new_labels:
                save_crop(image_path, f"obj_{idx}", crop_img, new_labels, output_images_dir, output_labels_dir, debug_dir, debug_limit_counter, class_names)
                generated_files.append(f"obj_{idx}")

        # --- B. Background Crops (Empty Images Only) ---
        if num_bg_crops > 0 and not objects:
            import random
            
            # Select centroids
            chosen_centroids = []
            if all_centroids and len(all_centroids) > 0:
                # Randomly sample from existing centroids
                # But we can't just slice, we need random choice
                # Since all_centroids might be huge, and we only need k, random.choices is good
                chosen_centroids = random.choices(all_centroids, k=num_bg_crops)
            else:
                # Fallback: Center of image
                chosen_centroids = [(0.5, 0.5)] * num_bg_crops
            
            for i, (cx_norm, cy_norm) in enumerate(chosen_centroids):
                abs_cx = cx_norm * img_w
                abs_cy = cy_norm * img_h
                
                crop_x1, crop_y1 = get_valid_crop_window(img_w, img_h, abs_cx, abs_cy, CROP_SIZE)
                crop_x2 = crop_x1 + CROP_SIZE
                crop_y2 = crop_y1 + CROP_SIZE
                
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Should have NO labels since image is empty
                # We force save it with empty labels
                save_crop(image_path, f"background_{i}", crop_img, [], output_images_dir, output_labels_dir, debug_dir, debug_limit_counter, class_names)
                generated_files.append(f"background_{i}")

        return generated_files

    except Exception as e:
        return [f"Error processing {image_path}: {str(e)}"]

def recalculate_labels(objects, crop_x1, crop_y1, img_w, img_h):
    # Extracted logic for cleanliness
    crop_x2 = crop_x1 + CROP_SIZE
    crop_y2 = crop_y1 + CROP_SIZE
    new_labels = []
    
    for obj in objects:
        ox_c = obj['x_c'] * img_w
        oy_c = obj['y_c'] * img_h
        ow = obj['w'] * img_w
        oh = obj['h'] * img_h
        
        ox1 = ox_c - ow / 2
        oy1 = oy_c - oh / 2
        ox2 = ox_c + ow / 2
        oy2 = oy_c + oh / 2
        
        ix1 = max(crop_x1, ox1)
        iy1 = max(crop_y1, oy1)
        ix2 = min(crop_x2, ox2)
        iy2 = min(crop_y2, oy2)
        
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            original_area = ow * oh
            
            if original_area > 0 and (inter_area / original_area) >= MIN_VISIBILITY_RATIO:
                new_x1 = ix1 - crop_x1
                new_y1 = iy1 - crop_y1
                new_x2 = ix2 - crop_x1
                new_y2 = iy2 - crop_y1
                
                new_w_abs = new_x2 - new_x1
                new_h_abs = new_y2 - new_y1
                new_cx_abs = new_x1 + new_w_abs / 2
                new_cy_abs = new_y1 + new_h_abs / 2
                
                norm_x = min(max(new_cx_abs / CROP_SIZE, 0.0), 1.0)
                norm_y = min(max(new_cy_abs / CROP_SIZE, 0.0), 1.0)
                norm_w = min(max(new_w_abs / CROP_SIZE, 0.0), 1.0)
                norm_h = min(max(new_h_abs / CROP_SIZE, 0.0), 1.0)
                
                new_labels.append(f"{obj['class_id']} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
    return new_labels

def save_crop(image_path, suffix, crop_img, labels, output_images_dir, output_labels_dir, debug_dir, debug_limit_counter, class_names):
    base_name = image_path.stem
    out_name = f"{base_name}_{suffix}"
    
    out_img_path = output_images_dir / f"{out_name}.jpg"
    out_lbl_path = output_labels_dir / f"{out_name}.txt"
    
    crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img_path), crop_bgr)
    
    with open(out_lbl_path, 'w') as f_out:
        f_out.write('\n'.join(labels))

    # Debug Visualization
    if len(debug_limit_counter) < 20: # Increased limit casually
        debug_limit_counter.append(1)
        
        debug_img = crop_bgr.copy()
        h_crop, w_crop = debug_img.shape[:2]
        
        if not labels:
             cv2.putText(debug_img, "Background / Empty", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for line in labels:
            parts = line.split()
            cls_id = int(parts[0])
            nx, ny, nw, nh = map(float, parts[1:])
            
            dx = int(nx * w_crop)
            dy = int(ny * h_crop)
            dw = int(nw * w_crop)
            dh = int(nh * h_crop)
            
            l = int(dx - dw/2)
            t = int(dy - dh/2)
            r = int(dx + dw/2)
            b = int(dy + dh/2)
            
            cv2.rectangle(debug_img, (l, t), (r, b), (0, 255, 0), 2)
            
            label_text = f"{cls_id}"
            if class_names and cls_id in class_names:
                label_text = f"{cls_id}: {class_names[cls_id]}"
                
            cv2.putText(debug_img, label_text, (l, t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        debug_out_path = debug_dir / f"DEBUG_{out_name}.jpg"
        cv2.imwrite(str(debug_out_path), debug_img)

if __name__ == "__main__":
    main()
