# This script is preprocessing process if dataset is in COCO format

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SOURCE_IMG = Path(r"rawdata/images")
SOURCE_JSON = Path(r"rawdata/annotation")
OUT_BASE = Path(r"datasets")

OUT_TRAIN_IMG = OUT_BASE / "images/train"
OUT_VAL_IMG   = OUT_BASE / "images/val"
OUT_TEST_IMG  = OUT_BASE / "images/test"
OUT_TRAIN_LBL = OUT_BASE / "labels/train"
OUT_VAL_LBL   = OUT_BASE / "labels/val"
OUT_TEST_LBL  = OUT_BASE / "labels/test"

for p in [OUT_TRAIN_IMG, OUT_VAL_IMG, OUT_TRAIN_LBL, OUT_VAL_LBL, OUT_TEST_IMG, OUT_TEST_LBL]:
    p.mkdir(parents=True, exist_ok=True)

CLASSES = {
    'dent': 0,
    'paint scratch': 1,
    'crack': 2,
    'broken glass': 3,
    'broken lamp': 4,
    'flat tire': 5
}

def clip(x):
    return max(0, min(1, round(x, 6)))

def save_yolo_label(lbl_path, lines):
    lbl_path.write_text("\n".join(lines), encoding="utf-8")

def make_unique_name(existing_names, file_name):
    stem = Path(file_name).stem
    ext = Path(file_name).suffix
    new_stem = stem
    count = 1
    while new_stem + ext in existing_names:
        new_stem = f"{stem}_{count}"
        count += 1
    existing_names.add(new_stem + ext)
    return new_stem, new_stem + ext

def json_to_yolo_lines(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not all(k in data for k in ["images", "annotations", "categories"]):
        return {}

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_info = {img["id"]: img for img in data["images"]}
    img_lines = defaultdict(list)

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_info or "bbox" not in ann:
            continue
        img_meta = img_info[img_id]
        w, h = img_meta["width"], img_meta["height"]
        img_file = img_meta["file_name"]

        x_min, y_min, w_box, h_box = ann["bbox"]
        x_center = clip((x_min + w_box/2)/w)
        y_center = clip((y_min + h_box/2)/h)
        w_norm = clip(w_box / w)
        h_norm = clip(h_box / h)

        cls_name = cat_map.get(ann["category_id"], None)
        if cls_name not in CLASSES:
            continue

        line = f"{CLASSES[cls_name]} {x_center} {y_center} {w_norm} {h_norm}"
        img_lines[img_file].append(line)

    return img_lines

# Step 1: Process JSONs to collect all images and labels
img_list = []  # (image_path, label_lines, class_name)

json_files = list(SOURCE_JSON.glob("*.json"))
print("Processing JSONs...")
for js in tqdm(json_files):
    img_lines_map = json_to_yolo_lines(js)
    for img_file, lines in img_lines_map.items():
        src_img = SOURCE_IMG / img_file
        if not src_img.exists():
            continue
        
        first_cls = int(float(lines[0].split()[0]))
        cls_name = [k for k,v in CLASSES.items() if v==first_cls][0]
        img_list.append((src_img, lines, cls_name))

print(f"Total images after rename: {len(img_list)}")

# Step 2: Stratified split 70:30
train_items, val_items = train_test_split(img_list, test_size=0.3, stratify=[item[2] for item in img_list], random_state=42)
val_items, test_items = train_test_split(val_items, test_size=0.5, stratify=[item[2] for item in val_items], random_state=42)

# Step 3: Save images and labels
def save_dataset(items, out_img_dir, out_lbl_dir):
    count = 1
    for src_img, lines, cls_name in tqdm(items):
        shutil.copy(src_img, out_img_dir / f"{count}{src_img.suffix}")
        lbl_file = out_lbl_dir / f"{count}.txt"
        save_yolo_label(lbl_file, lines)
        count += 1

print("Saving train dataset...")
save_dataset(train_items, OUT_TRAIN_IMG, OUT_TRAIN_LBL)
print("Saving val dataset...")
save_dataset(val_items, OUT_VAL_IMG, OUT_VAL_LBL)
print("Saving test dataset...")
save_dataset(test_items, OUT_TEST_IMG, OUT_TEST_LBL)

print(f"Train images: {len(train_items)}, Val images: {len(val_items)}, Test images: {len(test_items)}")