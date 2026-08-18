import os
from pathlib import Path
import pandas as pd
import shutil


def load_metadata(metadata_csv="data/metadata/dataset_info.csv"):
    return pd.read_csv(metadata_csv)


def stratified_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    import numpy as np
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    
    df_train_list, df_val_list, df_test_list = [], [], []
    
    # Split each source independently to honor domain requirements (e.g. prioritizing self_collected for test)
    for source, group_df in df.groupby('source'):
        src_name = Path(source).name.lower()
        if 'self' in src_name or 'collect' in src_name:
            # Prioritize test set for self-collected TP.HCM images: e.g. 20% train, 10% val, 70% test
            r_train, r_val, r_test = 0.20, 0.10, 0.70
            print(f"Applying test-heavy split (20/10/70) for self-collected source: {source}")
        else:
            r_train, r_val, r_test = train_size, val_size, test_size
            print(f"Applying standard split ({train_size}/{val_size}/{test_size}) for source: {source}")
            
        # Robust stratified split per label
        for label, label_df in group_df.groupby('label'):
            n = len(label_df)
            indices = label_df.index.tolist()
            
            # Shuffle using numpy random generator with seed
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
            
            n_train = max(1, int(round(r_train * n))) if r_train > 0 and n >= 1 else 0
            n_val = int(round(r_val * n)) if r_val > 0 else 0
            
            # Ensure n_train + n_val doesn't exceed total count
            if n_train + n_val > n:
                n_train = n - n_val
                if n_train < 0:
                    n_train = 0
                    n_val = n
            
            n_test = n - n_train - n_val
            
            df_train_list.append(group_df.loc[indices[:n_train]])
            df_val_list.append(group_df.loc[indices[n_train:n_train+n_val]])
            df_test_list.append(group_df.loc[indices[n_train+n_val:]])
            
    df_train = pd.concat(df_train_list, ignore_index=True) if df_train_list else pd.DataFrame(columns=df.columns)
    df_val = pd.concat(df_val_list, ignore_index=True) if df_val_list else pd.DataFrame(columns=df.columns)
    df_test = pd.concat(df_test_list, ignore_index=True) if df_test_list else pd.DataFrame(columns=df.columns)
    
    return df_train, df_val, df_test


def copy_to_processed(df_split, out_root="data/processed", subset_name="train"):
    out_root = Path(out_root)
    for _, row in df_split.iterrows():
        src = Path(row['path'])
        label = row['label']
        dest_dir = out_root / subset_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Avoid filename collisions by prefixing with the source folder name
        source_prefix = Path(row['source']).name
        new_filename = f"{source_prefix}_{src.name}"
        try:
            shutil.copy2(src, dest_dir / new_filename)
        except Exception as e:
            print(f"Failed to copy {src} to {dest_dir / new_filename}: {e}")


def main():
    meta_path = Path("data/metadata/dataset_info.csv")
    if not meta_path.exists():
        print("Metadata not found. Run clean_data.py first to generate dataset_info.csv")
        return
    df = load_metadata(str(meta_path))
    # keep only non-corrupt and non-duplicate images
    df = df[(df['is_corrupt'] == False) & (df.get('is_duplicate', False) == False)].reset_index(drop=True)
    
    # Apply unified label mapping if mapping CSV exists
    mapping_path = Path("data/metadata/label_mapping.csv")
    if mapping_path.exists():
        print("Loading label mapping from data/metadata/label_mapping.csv...")
        mapping_df = pd.read_csv(mapping_path)
        mapping_dict = {}
        for _, row in mapping_df.iterrows():
            dataset_name = row['source_dataset']
            orig_lbl = row['original_label']
            unif_lbl = row['unified_label']
            mapping_dict[(dataset_name, orig_lbl)] = unif_lbl
            
        mapped_labels = []
        for idx, row in df.iterrows():
            dataset_name = Path(row['source']).name
            orig_lbl = row['label']
            mapped_lbl = mapping_dict.get((dataset_name, orig_lbl), orig_lbl)
            mapped_labels.append(mapped_lbl)
        df['label'] = mapped_labels
        print("Successfully mapped source-specific labels to unified labels.")
    
    df_train, df_val, df_test = stratified_split(df)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    copy_to_processed(df_train, subset_name="train")
    copy_to_processed(df_val, subset_name="val")
    copy_to_processed(df_test, subset_name="test")
    print("Split complete. Files copied to data/processed/")


if __name__ == "__main__":
    main()
