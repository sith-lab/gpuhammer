import os
import shutil

HAMMER_ROOT = os.environ['HAMMER_ROOT']
val_dir = f"{HAMMER_ROOT}/results/fig12_t4/val"  # Directory containing flat list of .JPEG files
label_file = f"{HAMMER_ROOT}/results/fig12_t4/rand_val_labels.txt"

# Step 1: Read label file into a dict: filename -> synset
desired_files = {}
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        synset, fname = parts
        desired_files[fname] = synset

# Step 2: List all files currently in val/
all_files = {f for f in os.listdir(val_dir) if f.endswith(".JPEG") or f.endswith(".jpg")}

# Step 3: Move valid files to their corresponding folders
moved_count = 0
for fname, synset in desired_files.items():
    src_path = os.path.join(val_dir, fname)
    if not os.path.exists(src_path):
        print(f"⚠️ Warning: {fname} not found in val/ folder.")
        continue

    dst_dir = os.path.join(val_dir, synset)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, fname)

    shutil.move(src_path, dst_path)
    moved_count += 1

print(f"✅ Moved {moved_count} files into their synset folders.")

# Step 4: Delete unlisted files in the root val/ folder
deleted_count = 0
for fname in all_files:
    if fname not in desired_files:
        file_path = os.path.join(val_dir, fname)
        os.remove(file_path)
        deleted_count += 1
        print(f"🗑️ Deleted: {fname}")

print(f"🧹 Removed {deleted_count} unwanted files.")
