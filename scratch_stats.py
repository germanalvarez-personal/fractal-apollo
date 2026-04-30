import sys
sys.path.append("/home/jupyter/fractal-apollo")
from yolo_analyzer import YoloRustAnalyzer
import polars as pl
import os

# Initialize analyzer on their dataset
dataset_path = "/home/jupyter/datasets/dataset_porticos_v14/dataset_porticos_v14"
analyzer = YoloRustAnalyzer(dataset_path)

print("Loading data...")
# Load data lazily
lazy_df = analyzer.loader.load_data()

# Calculate area_rel and stats manually
q = lazy_df.with_columns([
    (pl.col("width") * pl.col("height")).alias("area_rel")
])

q = q.with_columns([
    pl.count("class_id").over("file_path").alias("object_count")
])

# Compute IQR bounds per class
q = q.with_columns([
    pl.col("area_rel").quantile(0.25).over("class_id").alias("area_q1"),
    pl.col("area_rel").quantile(0.75).over("class_id").alias("area_q3"),
    pl.count("area_rel").over("class_id").alias("cls_count")
])

q = q.with_columns(
    (pl.col("area_q3") - pl.col("area_q1")).alias("area_iqr")
)

# Collect to dataframe
df = q.select(["class_id", "area_q1", "area_q3", "area_iqr", "cls_count"]).unique().collect()
df = df.filter(pl.col("cls_count") >= 15)

print("\n--- IQR Statistics Per Class ---")
low_map = {}
high_map = {}
abs_tiny_map = {}
abs_oversized_map = {}

for row in df.iter_rows(named=True):
    cid = row["class_id"]
    q1 = row["area_q1"]
    q3 = row["area_q3"]
    iqr = row["area_iqr"]
    count = row["cls_count"]
    
    # Heuristic for area_iqr_low: floor should be 20% of Q1
    # Q1 - k * IQR = 0.2 * Q1  =>  0.8 * Q1 = k * IQR  =>  k = (0.8 * Q1) / IQR
    if iqr > 0:
        recommended_low = (0.8 * q1) / iqr
    else:
        recommended_low = 1.5 # default if no variance
        
    # Heuristic for area_iqr_high: ceiling should be 2.0x Q3? Or just standard 1.5?
    # Actually, let's keep 1.5x IQR as standard unless it goes over 1.0. 
    # Q3 + k * IQR = 1.0 => k = (1.0 - Q3) / IQR
    # But usually 1.5 is fine for high bounds
    recommended_high = 1.5
    
    # Store in maps
    low_map[cid] = round(recommended_low, 3)
    high_map[cid] = round(recommended_high, 3)
    
    # Also log what the actual floor/ceil will be
    floor_area = q1 - (recommended_low * iqr)
    ceil_area = q3 + (recommended_high * iqr)
    
    # Calculate absolute floors and ceilings for the metric maps
    abs_tiny_map[cid] = round(max(0.000001, floor_area), 6)
    abs_oversized_map[cid] = round(min(1.0, ceil_area), 6)

    print(f"Class {cid} (count={count}):")
    print(f"  Q1: {q1:.6f} | Q3: {q3:.6f} | IQR: {iqr:.6f}")
    print(f"  -> recommended low_multiplier  = {recommended_low:.3f} (Lower bound area: {floor_area:.6f})")
    print(f"  -> recommended high_multiplier = {recommended_high:.3f} (Upper bound area: {ceil_area:.6f})\n")

print("--------------------------------------------------")
print("Recommended Config Overrides for dataset_checker/config.py:")
print("--------------------------------------------------")
print("self.area_iqr_low_map = {")
for k, v in low_map.items():
    print(f"    {k}: {v},")
print("}")
print()
print("self.area_iqr_high_map = {")
for k, v in high_map.items():
    print(f"    {k}: {v},")
print("}")
print()
print("self.tiny_object_area_map = {")
for k, v in abs_tiny_map.items():
    print(f"    {k}: {v},")
print("}")
print()
print("self.oversized_safety_floor_map = {")
for k, v in abs_oversized_map.items():
    print(f"    {k}: {v},")
print("}")

