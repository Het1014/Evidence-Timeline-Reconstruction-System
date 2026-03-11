import json
import os

MANIFEST = "data/splits/selected_videos.json"  # your JSON file
RAW_DIR = "data/raw"

with open(MANIFEST, "r") as f:
    data = json.load(f)

# collect allowed absolute paths
allowed = set()
for videos in data.values():
    for v in videos:
        allowed.add(os.path.normpath(v["path"]))

# walk through raw dir
for root, _, files in os.walk(RAW_DIR):
    for file in files:
        if file.endswith(".mp4"):
            full_path = os.path.normpath(os.path.join(root, file))
            if full_path not in allowed:
                os.remove(full_path)
                print("Deleted:", full_path)