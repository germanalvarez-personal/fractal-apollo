import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import kornia_rs
from pathlib import Path
import math
import concurrent.futures
from .config import DatasetConfig
from .data_loader import DataLoader
from functools import partial

class Visualizer:
    def __init__(self, config: DatasetConfig, data_loader: DataLoader):
        self.config = config
        self.loader = data_loader # Need loader for path resolution
        
    def generate_heatmap(self, df: pl.DataFrame, bins=100, class_id=None, area_max=None) -> np.ndarray:
        """
        High-performance Heatmap generation using Polars binning.
        """
        # Filter
        subset = df
        if class_id is not None:
             subset = subset.filter(pl.col("class_id") == class_id)
        if area_max is not None:
             subset = subset.filter(pl.col("area_rel") < area_max)
             
        if subset.is_empty():
            return np.zeros((bins, bins))

        # Use Polars to aggregate into bins directly
        # x_bin = (x * bins).cast(int)
        # Group by x_bin, y_bin -> count
        
        heatmap_df = subset.select([
            (pl.col("x_center") * bins).cast(pl.Int32).clip(0, bins-1).alias("x_bin"),
            (pl.col("y_center") * bins).cast(pl.Int32).clip(0, bins-1).alias("y_bin")
        ]).group_by(["x_bin", "y_bin"]).count()
        
        # Convert to numpy matrix
        hm = np.zeros((bins, bins))
        
        # Iterate rows (much fewer than raw data)
        for row in heatmap_df.iter_rows():
            x, y, c = row
            hm[x, y] = c
            
        return hm.T

    def plot_dashboard(self, df: pl.DataFrame, total_images: int, save_path="dashboard.png"):
        print("Generating Dashboard...")
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle("YOLO Dataset Diagnostic Suite", fontsize=20)
        
        # 1. Class Frequency (Aggregated in Polars)
        class_counts = df.group_by("class_id").count().sort("count", descending=True).limit(20)
        pdf = class_counts.to_pandas()
        
        # Map names
        if self.loader.class_names:
            pdf["name"] = pdf["class_id"].map(lambda x: self.loader.class_names.get(x, str(x)))
            y_data = pdf["name"]
        else:
            y_data = pdf["class_id"].astype(str)
            
        sns.barplot(x=pdf["count"], y=y_data, ax=axes[0, 0], palette="viridis", orient='h')
        axes[0, 0].set_title("Top 20 Class Frequency")
        
        # 2. Global Heatmap
        hm = self.generate_heatmap(df, bins=100)
        sns.heatmap(hm, ax=axes[0, 1], cmap="inferno")
        axes[0, 1].set_title("Global Heatmap")
        axes[0, 1].invert_yaxis()
        
        # 4. Area Distribution (Log-Scale Box Plot)
        top_10 = class_counts.limit(10)["class_id"].to_list()
        # We need more data for boxplot to show distribution well, but not all of it for stripplot
        box_data = df.filter(pl.col("class_id").is_in(top_10)).select(["class_id", "area_rel"]).to_pandas()
        
        if self.loader.class_names:
             box_data["name"] = box_data["class_id"].map(lambda x: self.loader.class_names.get(x, str(x)))
             x_data = "name"
        else:
             x_data = "class_id"
             
        # Box plot (Main distribution)
        sns.boxplot(data=box_data, x=x_data, y="area_rel", ax=axes[0, 2], fliersize=0) # fliersize=0 because we overlay strip
        
        # Strip plot (Outliers/Details) - Sample if too large to prevent clutter/lag
        if len(box_data) > 2000:
             strip_data = box_data.sample(n=2000, random_state=42)
        else:
             strip_data = box_data
             
        sns.stripplot(data=strip_data, x=x_data, y="area_rel", ax=axes[0, 2], 
                      color="k", alpha=0.3, size=2, jitter=True)
                      
        axes[0, 2].set_title("Area Distribution (Log Scale)")
        axes[0, 2].set_yscale("log")
        axes[0, 2].axhline(y=self.config.tiny_object_area, color='r', linestyle='--', label="Tiny Thresh")
        axes[0, 2].legend()
        
        # 4. Hexbin (Sampled if too large?)
        # For hexbin we need raw data, but we can sample if > 100k to save time? 
        # Polars sample is fast.
        plot_df = df.select(["area_rel", "aspect_ratio", "edge_prox"]).sample(n=min(50000, len(df))).to_pandas()
        hb = axes[1, 0].hexbin(plot_df["area_rel"], plot_df["aspect_ratio"], gridsize=50, cmap='Blues', mincnt=1, xscale='log', yscale='log')
        axes[1, 0].set_title("Shape Analysis")
        fig.colorbar(hb, ax=axes[1, 0])
        
        # 5. Edge Bias
        sns.histplot(data=plot_df, x="edge_prox", bins=50, ax=axes[1, 1], kde=True)
        axes[1, 1].set_title("Edge Bias")
        
        # 6. Integrity Table
        flags = ["is_tiny", "is_out_of_bounds", "is_stretched", "is_duplicate", "is_inverted", "is_high_overlap"]
        report = []
        
        # Calc BG
        labeled_imgs = df["file_path"].n_unique()
        total_imgs = max(total_images, labeled_imgs)
        bg = total_imgs - labeled_imgs
        bg_pct = (bg / total_imgs * 100) if total_imgs > 0 else 0
        report.append(["Background Images", f"{bg}/{total_imgs}", f"{bg_pct:.2f}%"])
        
        for f in flags:
            if f in df.columns:
                c = df[f].sum()
                pct = (c / len(df) * 100)
                report.append([f, int(c), f"{pct:.2f}%"])
                
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=report, colLabels=["Metric", "Count", "%"], loc='center')
        table.scale(1, 1.5)
        axes[1, 2].set_title("Data Integrity")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Dashboard saved to {save_path}")

    def generate_stratified_mosaic(self, df: pl.DataFrame, save_path="stratified_mosaic.png"):
        print("Generating Parallel Stratified Mosaic...")
        
        # 1. Select instances
        class_counts = df.group_by("class_id").count()
        valid_classes = class_counts.filter(pl.col("count") >= 4)["class_id"].to_list()
        valid_classes.sort()
        
        # Limit to 40 classes for sanity
        valid_classes = valid_classes[:40]
        
        # Gather Tasks
        tasks = []
        
        for i, cls_id in enumerate(valid_classes):
             cls_df = df.filter(pl.col("class_id") == cls_id)
             # Get smallest and largest
             s = cls_df.sort("area_rel").head(2).to_dicts()
             l = cls_df.sort("area_rel", descending=True).head(2).to_dicts()
             instances = s + l
             
             for j, item in enumerate(instances):
                 tasks.append({
                     "cls_id": cls_id, 
                     "cls_idx": i, 
                     "item_idx": j, 
                     "data": item
                 })
                 
        # Layout
        clusters_per_row = 5
        rows = math.ceil(len(valid_classes) / clusters_per_row)
        ts = self.config.mosaic_tile_size
        mw = clusters_per_row * 2 * ts
        mh = rows * 2 * ts
        
        mosaic = Image.new("RGB", (mw, mh), (255, 255, 255))
        
        # Parallel Process
        # We need a partial function designed for the worker
        # But KorniaRS read should happen in the worker.
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # map returns iterator in order
            futures = [executor.submit(self._process_tile, task, ts) for task in tasks]
            
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    if res:
                        # Paste result
                        # Calc position
                        # Cluster pos
                        cid = res['task']['cls_idx']
                        j = res['task']['item_idx']
                        
                        r = cid // clusters_per_row
                        c = cid % clusters_per_row
                        
                        base_x = c * 2 * ts
                        base_y = r * 2 * ts
                        
                        tr = j // 2
                        tc = j % 2
                        
                        tx = base_x + tc * ts
                        ty = base_y + tr * ts
                        
                        mosaic.paste(res['img'], (tx, ty))
                        
                        # Label on first
                        if j == 0:
                            d = ImageDraw.Draw(mosaic)
                            name = self.loader.class_names.get(res['task']['cls_id'], str(res['task']['cls_id']))
                            d.text((tx+5, ty+5), f"{res['task']['cls_id']}: {name}", fill="red")
                            
                except Exception as e:
                    print(f"Tile error: {e}")
                    
        mosaic.save(save_path)
        print(f"Mosaic saved to {save_path}")

    def _process_tile(self, task, tile_size):
        # Worker function
        item = task['data']
        path_str = item['file_path']
        img_p = self.loader._resolve_image_path(Path(path_str))
        
        if not img_p.exists():
            return None
            
        # Read
        try:
            # Kornia RS is fast, but let's see if we can use it safely here
            # Ideally create new instance or use static?
            # It's a function import
            from kornia_rs import read_image_jpeg
            t = read_image_jpeg(str(img_p), "rgb")
            img_np = np.array(t)
            
            # HWC check
            if len(img_np.shape) == 3 and img_np.shape[0] == 3: # CHW?
                 img_np = np.transpose(img_np, (1, 2, 0))
            
            pil_img = Image.fromarray(img_np)
            
            # Crop
            iw, ih = pil_img.size
            xc, yc, w, h = item["x_center"], item["y_center"], item["width"], item["height"]
            
            x1 = int((xc - 0.5*w)*iw)
            y1 = int((yc - 0.5*h)*ih)
            x2 = int((xc + 0.5*w)*iw)
            y2 = int((yc + 0.5*h)*ih)
            
            pad_x = int((x2-x1)*0.25)
            pad_y = int((y2-y1)*0.25)
            
            x1 = max(0, x1-pad_x)
            y1 = max(0, y1-pad_y)
            x2 = min(iw, x2+pad_x)
            y2 = min(ih, y2+pad_y)
            
            crop = pil_img.crop((x1, y1, x2, y2))
            crop = crop.resize((tile_size, tile_size))
            
            # Box?
            # User wants clear bounding box.
            # We can draw it here on the crop before returning
            d = ImageDraw.Draw(crop)
            
            # Recalc box in crop
            # Original box coords
            ox1 = int((xc - 0.5*w)*iw)
            oy1 = int((yc - 0.5*h)*ih)
            ox2 = int((xc + 0.5*w)*iw)
            oy2 = int((yc + 0.5*h)*ih)
            
            # Scale to crop
            cw, ch = crop.size
            
            # map ox1 from [x1, x2] to [0, cw]
            # nx = (ox - x1) / (x2 - x1) * cw
            
            rw = x2 - x1
            rh = y2 - y1
            
            # protection div 0
            if rw > 0 and rh > 0:
                bx1 = (ox1 - x1) / rw * cw
                by1 = (oy1 - y1) / rh * ch
                bx2 = (ox2 - x1) / rw * cw
                by2 = (oy2 - y1) / rh * ch
                d.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
            
            return {"task": task, "img": crop}
            
        except Exception as e:
            # print(f"Error in worker: {e}")
            return None
