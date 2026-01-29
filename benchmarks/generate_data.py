
import os
import random
import time
from pathlib import Path
import numpy as np

def generate_dummy_data(base_path="data_test", num_files=100000, num_classes=80):
    images_dir = Path(base_path) / "images"
    labels_dir = Path(base_path) / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_files} dummy label files in {labels_dir}...")
    
    start_time = time.time()
    
    # We will write files in batches or parallel if needed, but simple loop might be fast enough for text
    # 100k files might take a bit.
    
    for i in range(num_files):
        # Random number of objects per image (1 to 10)
        num_objs = random.randint(1, 10)
        lines = []
        for _ in range(num_objs):
            cls = random.randint(0, num_classes - 1)
            xc = random.random()
            yc = random.random()
            w = random.uniform(0.01, 0.5)
            h = random.uniform(0.01, 0.5)
            # Clip to stay roughly in bounds (though some out of bounds is desired for testing flags)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        file_path = labels_dir / f"image_{i:06d}.txt"
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
            
    print(f"Generated {num_files} label files in {time.time() - start_time:.2f}s")

    print(f"Generated {num_files} label files in {time.time() - start_time:.2f}s")

    # Generate dummy images to test Background Ratio
    # We need total_images > num_files to have some background images
    # Let's create num_files + 1000 images.
    # We can just touch them for speed.
    print(f"Generating {num_files + 1000} dummy image files for background stats...")
    start_time = time.time()
    
    # Use a faster way to create files if possible, but loop is fine for 100k empty files
    # We create files corresponding to labels
    for i in range(num_files):
        (images_dir / f"image_{i:06d}.jpg").touch()
        
    # And background images (no corresponding label)
    for i in range(num_files, num_files + 1000):
        (images_dir / f"image_{i:06d}.jpg").touch()
        
    print(f"Generated images in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    generate_dummy_data()
