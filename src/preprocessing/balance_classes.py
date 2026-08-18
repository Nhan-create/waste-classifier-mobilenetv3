import os
from pathlib import Path
import collections
import json
import shutil


def compute_class_counts(processed_root="data/processed/train"):
    p = Path(processed_root)
    counts = {}
    for label_dir in p.iterdir():
        if label_dir.is_dir():
            counts[label_dir.name] = len(list(label_dir.glob("*.*")))
    return counts


def compute_class_weights(counts: dict):
    import numpy as np
    labels = list(counts.keys())
    freqs = np.array([counts[k] for k in labels], dtype=float)
    total = freqs.sum()
    weights = {labels[i]: float(total / (len(labels) * freqs[i])) for i in range(len(labels))}
    return weights


def oversample_copy(processed_root="data/processed/train", out_root="data/processed/train_oversampled"):
    counts = compute_class_counts(processed_root)
    max_count = max(counts.values())
    src_root = Path(processed_root)
    dst_root = Path(out_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for label, cnt in counts.items():
        src_dir = src_root / label
        dst_dir = dst_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = list(src_dir.glob("*.*"))
        # copy original files
        for f in files:
            shutil.copy2(f, dst_dir / f.name)
        # oversample by copying with new names
        if len(files) == 0:
            continue
        num_to_add = max_count - len(files)
        for i in range(num_to_add):
            src = files[i % len(files)]
            new_name = f"oversample_{i:05d}_{src.name}"
            shutil.copy2(src, dst_dir / new_name)


def main():
    counts = compute_class_counts()
    print("Class counts:", counts)
    weights = compute_class_weights(counts)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/class_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print("Saved class weights to data/processed/class_weights.json")
    print("To perform oversampling copy, run oversample_copy() or call oversample_copy function.")


if __name__ == "__main__":
    main()
