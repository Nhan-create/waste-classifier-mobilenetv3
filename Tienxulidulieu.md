# Tiền Xử Lý Dữ Liệu — Đồ Án Phân Loại Rác Thải (MobileNetV3)

## 1. Mục tiêu

Chuẩn bị và xử lý dữ liệu từ 3 nguồn (**VN Trash**, **Garbage Classification v2**, **ảnh tự chụp tại TP.HCM**) để huấn luyện mô hình phân loại rác **MobileNetV3**, đồng thời thiết kế các chiến lược **data augmentation** phục vụ mục tiêu so sánh thực nghiệm của đề tài (ảnh hưởng của augmentation đến hiệu suất mô hình trong điều kiện ánh sáng/góc chụp thực tế Việt Nam).

---

## 2. Cấu trúc thư mục dự án

```
waste-classification/
│
├── data/
│   ├── raw/                        # Dữ liệu gốc, chưa xử lý
│   │   ├── vn_trash/
│   │   ├── garbage_v2/
│   │   └── self_collected/
│   │
│   ├── interim/                    # Dữ liệu sau khi gộp + làm sạch, chưa augment
│   │   └── merged_clean/
│   │
│   ├── processed/                  # Dữ liệu sẵn sàng huấn luyện (đã chia train/val/test)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── metadata/
│       ├── label_mapping.csv       # Bảng ánh xạ nhãn giữa các nguồn
│       └── dataset_info.csv        # Metadata: đường dẫn, nhãn, nguồn, kích thước...
│
├── src/
│   └── preprocessing/
│       ├── merge_datasets.py       # Gộp 3 nguồn dữ liệu + áp label mapping
│       ├── clean_data.py           # Lọc ảnh lỗi, trùng lặp, nhiễu
│       ├── eda.py                  # Thống kê, trực quan hóa dữ liệu
│       ├── lighting_enhance.py     # CLAHE, gamma correction, white balance
│       ├── augmentation.py         # Định nghĩa 5 nhóm augmentation (A–E)
│       ├── split_dataset.py        # Chia stratified train/val/test
│       ├── balance_classes.py      # Xử lý mất cân bằng lớp (oversampling/weights)
│       └── dataloader.py           # PyTorch Dataset & DataLoader (transform pipeline)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Khám phá dữ liệu ban đầu
│   ├── 02_augmentation_test.ipynb  # Thử nghiệm trực quan các augmentation
│   └── 03_experiment_compare.ipynb # So sánh kết quả các chiến lược A–E
│
├── configs/
│   └── preprocessing_config.yaml   # Tham số: image size, mean/std, augmentation groups...
│
├── outputs/
│   ├── logs/                       # Log quá trình xử lý (số ảnh lỗi, trùng lặp...)
│   └── sample_visualizations/      # Ảnh minh họa trước/sau augmentation
│
└── README.md                       # File này
```

---

## 3. Các bước tiền xử lý (quy trình thực hiện)

| Bước | Nội dung | File thực hiện |
|---|---|---|
| 1 | Gộp 3 nguồn dữ liệu, thống nhất nhãn & cấu trúc thư mục | `merge_datasets.py` |
| 2 | Làm sạch: loại ảnh lỗi, trùng lặp (pHash), ảnh mờ/nhiễu | `clean_data.py` |
| 3 | Phân tích khám phá dữ liệu (phân bố lớp, nguồn, độ sáng) | `eda.py`, `01_eda.ipynb` |
| 4 | Tăng cường điều kiện ánh sáng thực tế (CLAHE, gamma...) | `lighting_enhance.py` |
| 5 | Thiết kế 5 nhóm augmentation (A: baseline → E: nền phức tạp) | `augmentation.py`, `02_augmentation_test.ipynb` |
| 6 | Chia tập Train/Val/Test (stratified, ưu tiên ảnh tự chụp cho test) | `split_dataset.py` |
| 7 | Xử lý mất cân bằng lớp (oversampling / class weights) | `balance_classes.py` |
| 8 | Xây dựng pipeline transform cho huấn luyện & inference (đồng nhất) | `dataloader.py` |

---

## 4. Danh sách file cần tạo

### 4.1 Dữ liệu & metadata
- [x] `data/metadata/label_mapping.csv`
- [ ] `data/metadata/dataset_info.csv`

### 4.2 Script tiền xử lý (`src/preprocessing/`)
- [x] `merge_datasets.py`
- [x] `clean_data.py`
- [x] `eda.py`
- [x] `lighting_enhance.py`
- [x] `augmentation.py`
- [x] `split_dataset.py`
- [x] `balance_classes.py`
- [x] `dataloader.py`

### 4.3 Notebook
- [ ] `notebooks/01_eda.ipynb`
- [ ] `notebooks/02_augmentation_test.ipynb`
- [ ] `notebooks/03_experiment_compare.ipynb`

### 4.4 Cấu hình & log
- [ ] `configs/preprocessing_config.yaml`
- [ ] `outputs/logs/cleaning_report.txt`
- [ ] `outputs/sample_visualizations/` (ảnh minh họa)

---

## 5. Ghi chú quan trọng

- **Augmentation chỉ áp dụng cho tập Train.** Tập Val/Test chỉ resize + normalize để đánh giá khách quan.
- **Test set nên ưu tiên ảnh tự chụp tại TP.HCM** để phản ánh đúng "điều kiện thực tế" mà đề tài hướng tới.
- **Pipeline tiền xử lý lúc inference (demo web)** phải giống hệt pipeline lúc validate, tránh lệch phân phối dữ liệu (data distribution shift).
- Giữ lại ảnh gốc (chưa augment) để phục vụ **Grad-CAM / heatmap** trong ứng dụng demo.
- Ghi log chi tiết ở mỗi bước làm sạch (số ảnh loại bỏ, lý do) để đưa vào báo cáo khóa luận.

---

## 6. Thứ tự chạy pipeline

```bash
python src/preprocessing/merge_datasets.py
python src/preprocessing/clean_data.py
python src/preprocessing/eda.py
python src/preprocessing/lighting_enhance.py
python src/preprocessing/split_dataset.py
python src/preprocessing/balance_classes.py
# augmentation.py và dataloader.py được import trực tiếp trong lúc huấn luyện
```
