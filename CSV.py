import os
import csv

geotiff_dir = "NRCC251128_output/GeoTIFFs"
csv_path = "NRCC251128_output/predictions.csv"

geotiff_files = sorted([f for f in os.listdir(geotiff_dir) if f.endswith(".tif")])

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "predicted_mask"])
    for tif_file in geotiff_files:
        image_id = tif_file.replace(".tif", "")
        writer.writerow([image_id, f"GeoTIFFs/{tif_file}"])

print(f"CSV created at: {csv_path}")