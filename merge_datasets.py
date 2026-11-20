import os
import shutil
from pathlib import Path

# ===== الإعدادات =====
CURRENT_DATASET = "dataset"
EXTRA_DATASET = "dataset_extra"
OUTPUT_DATASET = "dataset_merged"

# خريطة الفئات: extra_id → current_id
# current names: ['barricade', 'dumpster', 'excavators', 'gloves', 'hardhat', 'mask',
#                 'no-hardhat', 'no-mask', 'no-safety vest', 'person', 'safety net',
#                 'safety shoes', 'safety vest', 'dump truck', 'mini-van', 'truck', 'wheel loader']
# extra names: ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']

CLASS_MAP = {
    0: 4,    # helmet → hardhat
    1: 6,    # no-helmet → no-hardhat
    2: 8,    # no-vest → no-safety vest
    3: 9,    # person → person
    4: 12,   # vest → safety vest
}


def process_labels(src_labels_dir, dst_labels_dir, prefix="extra_"):
    """تحويل labels من extra إلى صيغة dataset"""
    os.makedirs(dst_labels_dir, exist_ok=True)
    count = 0
    
    for label_file in os.listdir(src_labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        src_path = os.path.join(src_labels_dir, label_file)
        new_name = prefix + label_file
        dst_path = os.path.join(dst_labels_dir, new_name)
        
        with open(src_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            if class_id not in CLASS_MAP:
                continue  # تجاهل الفئات غير المدعومة
            
            new_class_id = CLASS_MAP[class_id]
            new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
            new_lines.append(new_line)
        
        if new_lines:  # فقط إذا كان فيه detections
            with open(dst_path, 'w') as f:
                f.writelines(new_lines)
            count += 1
    
    return count


def copy_images(src_images_dir, dst_images_dir, prefix="extra_"):
    """نسخ الصور"""
    os.makedirs(dst_images_dir, exist_ok=True)
    count = 0
    
    for img_file in os.listdir(src_images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        src_path = os.path.join(src_images_dir, img_file)
        new_name = prefix + img_file
        dst_path = os.path.join(dst_images_dir, new_name)
        
        shutil.copy2(src_path, dst_path)
        count += 1
    
    return count


# ===== Main =====
print("=" * 60)
print("بدء دمج المجموعتين...")
print("=" * 60)

# مسح المجلد القديم إذا موجود
if os.path.exists(OUTPUT_DATASET):
    print(f"حذف المجلد القديم {OUTPUT_DATASET}...")
    shutil.rmtree(OUTPUT_DATASET)

for split in ['train', 'valid']:
    print(f"\n📁 معالجة {split}...")
    
    # مسارات المصدر
    current_images = os.path.join(CURRENT_DATASET, f"{split}_yolo", "images")
    current_labels = os.path.join(CURRENT_DATASET, f"{split}_yolo", "labels")
    extra_images = os.path.join(EXTRA_DATASET, split, "images")
    extra_labels = os.path.join(EXTRA_DATASET, split, "labels")
    
    # مسارات الهدف
    out_images = os.path.join(OUTPUT_DATASET, split, "images")
    out_labels = os.path.join(OUTPUT_DATASET, split, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    
    # نسخ صور المجموعة الحالية
    print(f"  - نسخ صور المجموعة الحالية...")
    for f in os.listdir(current_images):
        shutil.copy2(os.path.join(current_images, f), os.path.join(out_images, f))
    print(f"    ✅ {len(os.listdir(current_images))} صورة")
    
    # نسخ labels المجموعة الحالية
    if os.path.exists(current_labels):
        for f in os.listdir(current_labels):
            shutil.copy2(os.path.join(current_labels, f), os.path.join(out_labels, f))
        print(f"    ✅ {len(os.listdir(current_labels))} label")
    
    # نسخ وتحويل صور المجموعة الجديدة
    print(f"  - نسخ وتحويل المجموعة الجديدة...")
    n_imgs = copy_images(extra_images, out_images, prefix="extra_")
    print(f"    ✅ {n_imgs} صورة")

    n_labels = process_labels(extra_labels, out_labels, prefix="extra_")
    print(f"    ✅ {n_labels} label")

print("\n" + "=" * 60)
print("✅ اكتمل الدمج!")
print("=" * 60)

# الإحصائيات
for split in ['train', 'valid']:
    imgs = len(os.listdir(os.path.join(OUTPUT_DATASET, split, "images")))
    labels = len(os.listdir(os.path.join(OUTPUT_DATASET, split, "labels")))
    print(f"{split}: {imgs} صورة، {labels} label")