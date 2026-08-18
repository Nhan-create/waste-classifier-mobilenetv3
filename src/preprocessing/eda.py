import os
from pathlib import Path
import pandas as pd

def run_eda(metadata_csv="data/metadata/dataset_info.csv", output_dir="outputs/logs"):
    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        print(f"Metadata file not found at {meta_path}. Please run clean_data.py first.")
        return
        
    df = pd.read_csv(meta_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. General counts
    total_imgs = len(df)
    corrupt_imgs = df['is_corrupt'].sum()
    duplicate_imgs = df['is_duplicate'].sum() if 'is_duplicate' in df.columns else 0
    valid_imgs = total_imgs - corrupt_imgs - duplicate_imgs
    
    report = []
    report.append("="*50)
    report.append("DATASET EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    report.append("="*50)
    report.append(f"Total images scanned: {total_imgs}")
    report.append(f"Corrupt images:       {corrupt_imgs} ({corrupt_imgs/total_imgs*100:.2f}%)")
    report.append(f"Duplicate images:     {duplicate_imgs} ({duplicate_imgs/total_imgs*100:.2f}%)")
    report.append(f"Valid images:         {valid_imgs} ({valid_imgs/total_imgs*100:.2f}%)")
    report.append("-" * 50)
    
    # 2. Break down by source
    report.append("IMAGES BY SOURCE:")
    source_counts = df.groupby('source').size()
    for src, cnt in source_counts.items():
        src_name = Path(src).name
        corrupt_src = df[df['source'] == src]['is_corrupt'].sum()
        dup_src = df[df['source'] == src]['is_duplicate'].sum() if 'is_duplicate' in df.columns else 0
        report.append(f"  - {src_name}: {cnt} images (Corrupt: {corrupt_src}, Duplicate: {dup_src})")
    report.append("-" * 50)
    
    # 3. Class distributions for valid images
    valid_df = df[(df['is_corrupt'] == False) & (df.get('is_duplicate', False) == False)]
    report.append("CLASS DISTRIBUTION (VALID IMAGES ONLY):")
    class_counts = valid_df['label'].value_counts()
    for label, cnt in class_counts.items():
        pct = cnt / len(valid_df) * 100
        report.append(f"  - {label:<15}: {cnt:<5} ({pct:.2f}%)")
    report.append("-" * 50)
    
    # 4. Class distribution per source
    report.append("CLASS DISTRIBUTION BY SOURCE:")
    src_class = valid_df.groupby(['source', 'label']).size().unstack(fill_value=0)
    for src, row in src_class.iterrows():
        src_name = Path(src).name
        report.append(f"  Source: {src_name}")
        for label, cnt in row.items():
            if cnt > 0:
                report.append(f"    * {label:<12}: {cnt}")
    report.append("-" * 50)
    
    # 5. Image dimensions
    report.append("IMAGE RESOLUTION STATISTICS:")
    widths = valid_df['width'].dropna()
    heights = valid_df['height'].dropna()
    if len(widths) > 0 and len(heights) > 0:
        report.append(f"  Width  - Min: {widths.min():.0f}, Max: {widths.max():.0f}, Mean: {widths.mean():.1f}, Median: {widths.median():.0f}")
        report.append(f"  Height - Min: {heights.min():.0f}, Max: {heights.max():.0f}, Mean: {heights.mean():.1f}, Median: {heights.median():.0f}")
    else:
        report.append("  No dimension data available.")
    report.append("="*50)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(out_path / "eda_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved EDA report to {out_path / 'eda_report.txt'}")

if __name__ == "__main__":
    run_eda()
