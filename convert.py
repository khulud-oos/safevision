import json
import os
import shutil
from pathlib import Path

def convert_coco_to_yolo(coco_json, images_dir, output_dir):
    """Convert COCO annotations to YOLO format"""
    
    with open(coco_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output dirs
    output_images = os.path.join(output_dir, 'images')
    output_labels = os.path.join(output_dir, 'labels')
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    # Map category ids to yolo indices
    cat_id_to_yolo = {}
    for i, cat in enumerate(data['categories']):
        cat_id_to_yolo[cat['id']] = i
        print(f"Class {i}: {cat['name']}")
    
    # Group annotations by image
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image
    for img in data['images']:
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        file_name = img['file_name']
        
        # Copy image
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(output_images, file_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"Missing image: {file_name}")
            continue
        
        # Create label file
        label_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(output_labels, label_name)
        
        with open(label_path, 'w') as f:
            for ann in img_to_anns.get(img_id, []):
                bbox = ann['bbox']  # [x, y, w, h] in COCO
                x, y, w, h = bbox
                
                # Convert to YOLO format
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                cat_id = ann['category_id']
                yolo_cls = cat_id_to_yolo[cat_id]
                
                f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# === التطبيق ===
base = "dataset"
for split in ['train', 'valid']:
    coco_file = f"{base}/{split}/_annotations.coco.json"
    images_dir = f"{base}/{split}"
    output_dir = f"{base}/{split}_yolo"
    
    if os.path.exists(coco_file):
        print(f"\n=== Converting {split} ===")
        convert_coco_to_yolo(coco_file, images_dir, output_dir)
    else:
        print(f"Not found: {coco_file}")

print("\n✅ Conversion complete!")