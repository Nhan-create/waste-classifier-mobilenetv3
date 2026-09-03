# Tiền xử lý dữ liệu cho Waste Classifier 10 lớp

Tài liệu này mô tả đúng pipeline hiện tại. Pipeline dùng hai nguồn công khai, không có nguồn ảnh tự chụp và không ghi sẵn số lượng ảnh đầu ra. Số lượng phải được lấy từ report của chính lần chạy vì dataset, ảnh lỗi và duplicate audit quyết định kết quả.

## 1. Nguồn và bất biến

- VN Trash Classification: `mrgetshjtdone/vn-trash-classification/versions/1` — MIT.
- Garbage Classification v2: `sumn2u/garbage-classification-v2/versions/12` — MIT.
- Tập class ID theo đúng thứ tự: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`.
- Tỷ lệ mặc định: train `0.70`, validation `0.15`, test `0.15`; seed `42`.
- Train được augmentation; validation/test chỉ resize và normalize.
- Không một duplicate cluster nào được phép xuất hiện ở nhiều split.
- Mỗi split vật lý phải có đủ đúng 10 thư mục lớp.
- Pipeline chỉ tạo version mới và từ chối ghi đè `data/processed/v1` đã tồn tại.

Các giá trị trên nằm trong `configs/preprocessing_config.yaml`; thứ tự lớp nằm trong `src/data/schema.py`.

## 2. Ánh xạ nhãn

`data/metadata/label_mapping.csv` là bảng ánh xạ duy nhất cho VN Trash:

| Nhãn nguồn | Nhãn chuẩn |
|---|---|
| `Alu` | `metal` |
| `Carton` | `cardboard` |
| `Foam_box` | `plastic` |
| `Milk_box` | `cardboard` |
| `Other` | `trash` |
| `PET` | `plastic` |
| `Paper` | `paper` |
| `Paper_cup` | `paper` |
| `Plastic_cup` | `plastic` |

Garbage Classification v2 đã có 10 nhãn chuẩn; pipeline chỉ chuẩn hóa chữ thường, dấu cách và dấu gạch. Nhãn ngoài hợp đồng không được ghép theo phỏng đoán mà nhận trạng thái `unmapped`.

## 3. Luồng xử lý và thuật toán

### Bước 1 — Ingest có provenance

`src/preprocessing/merge_datasets.py` đọc layout `Train/Test` của VN Trash và layout `original` hoặc split/class của Garbage v2. Với mỗi file hợp lệ về phần mở rộng:

1. lưu `source_dataset`, `original_label`, `original_split` và `source_path`;
2. tạo `image_id` ổn định bằng SHA-256 của chuỗi provenance `source + split + relative_path`;
3. copy vào `data/raw/<source>/<label>/<image_id>.<ext>`;
4. nếu đường dẫn đích đã tồn tại nhưng bytes khác, dừng thay vì ghi đè.

Output là `raw_manifest.csv`; provenance không bị mất khi hợp nhất hai nguồn có tên file giống nhau.

### Bước 2 — Kiểm tra ảnh và tạo fingerprint

`src/preprocessing/clean_data.py` cùng `src/data/dedup.py`:

- Pillow mở, `load()` và chuyển thử ảnh sang RGB;
- ghi extension, width, height, mode;
- tính **SHA-256 của bytes** để nhận biết exact duplicate;
- tính pHash để nhận biết ảnh gần trùng theo khoảng cách Hamming, mặc định `<= 4`;
- dùng chỉ mục BK-tree để tìm hàng xóm pHash và union-find để gom các quan hệ trùng thành cluster bắc cầu.

Ảnh corrupt nhận lý do loại cụ thể. SHA-256 được dùng xuyên suốt; pipeline không dùng MD-5.

### Bước 3 — Xử lý duplicate và xung đột nhãn

Mỗi duplicate cluster chỉ giữ một đại diện xác định theo thứ tự ổn định. Nếu các thành viên trong cùng cluster ánh xạ sang nhiều class ID khác nhau, toàn cluster được đánh dấu xung đột/cách ly để review; pipeline không tự chọn nhãn “đa số”.

Các output audit:

- `scanned_manifest.csv` — trạng thái từng ảnh;
- `duplicate_clusters.csv` — exact/near duplicate;
- `label_conflicts.csv` — cluster có nhãn mâu thuẫn.

### Bước 4 — Chia group-stratified, deterministic

`src/data/split.py` chia độc lập theo bucket `source_dataset × unified_label`. Đơn vị chia là **cluster**, không phải file:

1. kiểm tra mỗi bucket có ít nhất ba cluster;
2. sắp cluster bằng khóa SHA-256 chứa seed;
3. cấp trước một cluster cho train, val và test;
4. gán các cluster còn lại vào split làm nhỏ nhất tổng sai lệch so với mục tiêu 70/15/15;
5. kiểm tra mỗi split có đúng tập 10 lớp.

Cùng manifest, mapping, tỷ lệ và seed sẽ cho cùng kết quả. Cách chia theo cluster ngăn bản sao hoặc ảnh gần trùng đi qua ranh giới split.

### Bước 5 — Materialize bất biến

Pipeline dựng cây tạm:

```text
data/processed/v1/
├── train/{10-class-ids}/
├── val/{10-class-ids}/
└── test/{10-class-ids}/
```

Chỉ khi mọi copy và manifest thành công, cây tạm mới được đổi tên thành version chính thức. Nếu có lỗi, cây build tạm được dọn; version đã tồn tại không bị sửa hoặc trộn âm thầm.

### Bước 6 — Validator chống leakage

`src/data/validation.py` không chỉ tin manifest. Nó quét lại file đã materialize, tính lại SHA-256/pHash và kiểm tra:

- đúng ba split và đúng 10 class directories;
- manifest khớp file vật lý;
- không exact duplicate đi qua split;
- không near-duplicate cluster đi qua split.

Huấn luyện gọi validator và dừng nếu `validation.is_valid` là false.

### Bước 7 — Class weights

Không oversample vật lý mặc định. `src/preprocessing/balance_classes.py` tính trọng số chỉ từ train:

```text
w_c = N / (K × n_c)
```

Trong đó `N` là tổng ảnh train, `K = 10`, `n_c` là số ảnh của lớp `c`. Trainer dựng tensor theo thứ tự class ID chuẩn, không phụ thuộc thứ tự key JSON.

### Bước 8 — Transform

Nhóm augmentation C mặc định cho train:

- `RandomResizedCrop(224, scale=(0.7, 1.0))`;
- horizontal flip;
- rotation tối đa 20 độ;
- color jitter;
- chuyển tensor và normalize ImageNet.

Validation, test và inference chỉ resize `224×224`, chuyển tensor và normalize bằng mean/std trong config/checkpoint. Không augment val/test để giữ phép đo ổn định.

## 4. Dataset fingerprint

Fingerprint checkpoint là SHA-256 của:

1. split manifest được sắp ổn định;
2. bytes của `label_mapping.csv`;
3. seed và tỷ lệ split.

Nhờ vậy có thể phát hiện khi model được dùng với một phiên bản dữ liệu hoặc mapping khác.

## 5. Output của một lần chạy

```text
data/metadata/v1/
├── raw_manifest.csv
├── scanned_manifest.csv
├── duplicate_clusters.csv
├── label_conflicts.csv
├── split_manifest.csv
└── class_weights.json
outputs/data/v1/
├── eda.json
└── eda_report.txt
```

`eda.json`/`eda_report.txt` ghi thống kê theo source, nhãn, trạng thái, split, lý do loại và dataset fingerprint. Dùng các file này để báo số liệu thực nghiệm; không sao chép số lượng từ một lần chạy cũ.

## 6. Lệnh chạy

```bash
python -m src.preprocessing.run_pipeline \
  --config configs/preprocessing_config.yaml \
  --raw-root data/raw \
  --metadata-root data/metadata/v1 \
  --processed-root data/processed/v1 \
  --report-root outputs/data/v1
```

Xác nhận lại độc lập:

```bash
python -m src.data.validation \
  --dataset-root data/processed/v1 \
  --manifest data/metadata/v1/split_manifest.csv \
  --phash-threshold 4
```

Lệnh thành công trả JSON chứa `dataset_fingerprint` và `zero_leakage: true`. Nếu muốn chạy lại, dùng version/output mới; không xóa một version đang được checkpoint tham chiếu mà chưa lưu manifest và cấu hình tương ứng.

## 7. Điều kiện trước khi huấn luyện

- `validation.is_valid == true`;
- đủ 10 lớp trong train/val/test;
- `split_manifest.csv`, mapping và config được giữ cùng experiment;
- class weights chỉ được tính từ train;
- không đọc test để chọn hyperparameter hoặc checkpoint.

Notebook `notebooks/train_mobilenetv3_kaggle.ipynb` tự động thực hiện các bước này và đóng gói manifest/config cùng `best.pt` để lần chạy có thể kiểm toán.
