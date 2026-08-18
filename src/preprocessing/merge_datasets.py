import os
import shutil
from pathlib import Path

def merge_vn_trash(src_root, dst_root):
    # VN Trash classification has Train and Test folders, inside them are Alu, Carton, Foam_box
    src_path = Path(src_root)
    dst_path = Path(dst_root) / "vn_trash"
    
    if not src_path.exists():
        print(f"Source VN Trash not found at {src_path}")
        return
        
    print("Merging VN Trash dataset...")
    count = 0
    # Traverse Train and Test splits
    for split in ["Train", "Test"]:
        split_dir = src_path / split
        if not split_dir.exists():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            target_dir = dst_path / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_path in label_dir.glob("*.*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                    # To avoid name collision, prefix with split name
                    new_name = f"{split.lower()}_{img_path.name}"
                    shutil.copy2(img_path, target_dir / new_name)
                    count += 1
    print(f"Copied {count} images for VN Trash to {dst_path}")

def merge_garbage_v2(src_root, dst_root):
    # Garbage Dataset has original folder, inside are battery, biological, cardboard, etc.
    src_path = Path(src_root) / "original"
    dst_path = Path(dst_root) / "garbage_v2"
    
    if not src_path.exists():
        print(f"Source Garbage Dataset not found at {src_path}")
        return
        
    print("Merging Garbage Dataset...")
    count = 0
    for label_dir in src_path.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        target_dir = dst_path / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for img_path in label_dir.glob("*.*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                shutil.copy2(img_path, target_dir / img_path.name)
                count += 1
    print(f"Copied {count} images for Garbage Dataset to {dst_path}")

def main():
    # Standard output raw directory
    dst_root = Path("data/raw")
    dst_root.mkdir(parents=True, exist_ok=True)
    
    # We will search in the 'Data' directory
    raw_source_root = Path("Data")
    if not raw_source_root.exists():
        print("Data directory not found. Please make sure the 'Data' directory exists in the workspace.")
        return
        
    # Merge VN Trash
    merge_vn_trash(raw_source_root / "VN Trash classification/VN_trash_classification", dst_root)
    
    # Merge Garbage Dataset
    merge_garbage_v2(raw_source_root / "Garbage Dataset", dst_root)
    
    print("Dataset merging complete. Now you can run: python src/preprocessing/clean_data.py")

if __name__ == "__main__":
    main()
