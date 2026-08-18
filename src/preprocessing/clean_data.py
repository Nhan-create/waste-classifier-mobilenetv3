import os
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash
import pandas as pd
from tqdm import tqdm


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(root_dirs, out_csv="data/metadata/dataset_info.csv"):
    rows = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for src in root_dirs:
        for label_dir in Path(src).iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_path in label_dir.rglob("*.*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in valid_extensions:
                    continue
                meta = {
                    "path": str(img_path).replace('\\\\', '/'),
                    "label": label,
                    "source": src,
                    "width": None,
                    "height": None,
                    "mode": None,
                    "md5": None,
                    "phash": None,
                    "is_corrupt": False,
                    "is_duplicate": False,
                }
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    with Image.open(img_path) as im:
                        meta["width"], meta["height"] = im.size
                        meta["mode"] = im.mode
                        # ensure RGB for perceptual hash
                        im_rgb = im.convert("RGB")
                        meta["phash"] = str(imagehash.phash(im_rgb))
                except (UnidentifiedImageError, OSError, ValueError):
                    meta["is_corrupt"] = True
                try:
                    meta["md5"] = md5_of_file(img_path)
                except Exception:
                    meta["md5"] = None

                rows.append(meta)

    df = pd.DataFrame(rows)
    # We will mark duplicates in the main function and save it there
    return df


def find_duplicates(df):
    # duplicates by md5
    dup_md5 = df[df.duplicated(subset=["md5"], keep=False) & df["md5"].notna()].sort_values("md5")
    # perceptual hash duplicates (exact phash)
    dup_phash = df[df.duplicated(subset=["phash"], keep=False) & df["phash"].notna()].sort_values("phash")
    return dup_md5, dup_phash


def main():
    # expect raw data in data/raw/* (folders per source)
    raw_root = Path("data/raw")
    if not raw_root.exists():
        print("No data/raw directory found — please prepare raw datasets under data/raw/")
        return
    sources = [str(p) for p in raw_root.iterdir() if p.is_dir()]
    print("Scanning sources:", sources)
    df = scan_dataset(sources)
    
    # Mark duplicates (keep only the first occurrence)
    if 'md5' in df.columns:
        df.loc[df.duplicated(subset=['md5'], keep='first') & df['md5'].notna(), 'is_duplicate'] = True
    if 'phash' in df.columns:
        df.loc[df.duplicated(subset=['phash'], keep='first') & df['phash'].notna(), 'is_duplicate'] = True

    # Save to standard path
    out_csv = "data/metadata/dataset_info.csv"
    out_dir = Path(out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    dup_md5, dup_phash = find_duplicates(df)
    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(logs_dir / "scan_dataset_full.csv", index=False)
    dup_md5.to_csv(logs_dir / "duplicates_md5.csv", index=False)
    dup_phash.to_csv(logs_dir / "duplicates_phash.csv", index=False)
    
    num_corrupt = df['is_corrupt'].sum()
    num_duplicates = df['is_duplicate'].sum()
    print(f"Scan complete. Total: {len(df)} images.")
    print(f"  - Corrupt: {num_corrupt}")
    print(f"  - Duplicates: {num_duplicates}")
    print(f"Reports saved in {logs_dir} and metadata saved to {out_csv}")


if __name__ == "__main__":
    main()
