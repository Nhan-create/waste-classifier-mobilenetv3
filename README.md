# Waste Classifier - MobileNetV3 (Tiền xử lý dữ liệu)

Dự án này là phân hệ tiền xử lý dữ liệu cho đồ án khóa luận tốt nghiệp về phân loại rác thải tại Việt Nam sử dụng mô hình mạng **MobileNetV3**.

Hệ thống được thiết kế để gộp, làm sạch và chuẩn bị dữ liệu từ 3 nguồn:
1. **VN Trash** (Bộ dữ liệu rác thải Việt Nam).
2. **Garbage Classification v2** (Bộ dữ liệu rác thải quốc tế).


---

## Những công việc đã thực hiện (What we did)

Chúng tôi đã hoàn thiện và sửa đổi toàn bộ pipeline tiền xử lý dữ liệu để đạt hiệu năng tối đa và đảm bảo tính chính xác cho việc huấn luyện mô hình:

### 1. Gộp và Chuẩn hóa cấu trúc dữ liệu (`merge_datasets.py`)
* Thiết kế mã nguồn gộp tự động dữ liệu từ thư mục tải về `Data/` sang cấu trúc chuẩn hóa `data/raw/`.
* Tự động xử lý cấu trúc phân cấp phức tạp (như phân chia sẵn `Train`/`Test` của VN Trash hoặc cấu trúc phẳng của Garbage Dataset) mà không gây nhầm lẫn nhãn.

### 2. Làm sạch & Loại bỏ trùng lặp (`clean_data.py`)
* Quét và kiểm tra tính toàn vẹn của tệp ảnh để lọc ảnh lỗi (corrupt).
* Sử dụng thuật toán băm **MD5** kết hợp băm cảm nhận **pHash (Perceptual Hash)** để phát hiện ảnh trùng lặp hoặc gần trùng lặp.
* **Loại bỏ trùng lặp triệt để** (giữ lại 1 bản ghi duy nhất) nhằm ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage) giữa tập huấn luyện (Train) và tập kiểm thử (Test).
* Thêm bộ lọc định dạng tệp ảnh chỉ cho phép định dạng ảnh hợp lệ (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`), bỏ qua các file rác của hệ thống (như `.DS_Store`).

### 3. Áp dụng bảng ánh xạ nhãn (`split_dataset.py`)
* Đọc cấu hình từ bảng ánh xạ [`label_mapping.csv`](data/metadata/label_mapping.csv).
* Chuyển đổi các nhãn riêng lẻ của từng bộ dữ liệu thành tập nhãn chung thống nhất (ví dụ: `Alu` -> `metal`, `Carton` -> `cardboard`, `PET` -> `plastic`, `Foam_box` -> `plastic`).

### 4. Chia tập dữ liệu tối ưu hóa cho Test set thực tế (`split_dataset.py`)
* Phân chia tập dữ liệu thành 3 phần **Train/Val/Test** theo cơ chế **Source-Aware Stratified Split** (chia độc lập theo từng nguồn):
  * **Bộ ảnh tự chụp (`self_collected`)**: Ưu tiên phân chia **20% Train / 10% Val / 70% Test** để dùng tập ảnh thực tế TP.HCM làm tập đánh giá chủ đạo.
  * **Các bộ dữ liệu công cộng khác**: Áp dụng tỷ lệ chuẩn **70% Train / 15% Val / 15% Test** để tối ưu hóa lượng dữ liệu huấn luyện.
* Khắc phục hoàn toàn lỗi crash khi chia dữ liệu nếu lớp có quá ít ảnh.
* Ngăn chặn ghi đè tệp tin trùng tên từ các nguồn khác nhau bằng cách tự động đánh số/tiền tố hóa tên tệp (`<source>_<filename>`).

### 5. Cân bằng lớp & Trọng số hàm Loss (`balance_classes.py`)
* Sửa lỗi nghiêm trọng về hiệu năng quét ổ đĩa liên tục ($O(N^2)$) khi sao chép oversampling.
* Tính toán tự động trọng số lớp nghịch đảo tần suất lớp để làm tham số cân bằng cho hàm loss (`nn.CrossEntropyLoss`) lưu tại `data/processed/class_weights.json`.

### 6. Cải tiến nhóm tăng cường dữ liệu (`augmentation.py` & `dataloader.py`)
* Loại bỏ các bước `Resize` lặp lại dư thừa trước `RandomResizedCrop` để tránh hiện tượng làm mờ ảnh và răng cưa ảnh.
* Tích hợp cấu hình 5 nhóm tăng cường dữ liệu thực nghiệm (A - Baseline đến E - Extreme) trực tiếp vào `get_transforms` của PyTorch `DataLoader` để tiện chạy các thực nghiệm so sánh.

---

## Kết quả chạy thực tế trên bộ dữ liệu

* **Tổng số ảnh quét ban đầu**: `15,754` ảnh.
* **Ảnh trùng lặp phát hiện & loại bỏ**: `923` ảnh.
* **Tổng ảnh hợp lệ đưa vào huấn luyện**: `14,831` ảnh.
* **Tỷ lệ phân chia tập dữ liệu**:
  * **Train**: `10,381` ảnh
  * **Validation**: `2,225` ảnh
  * **Test**: `2,225` ảnh
* **Phân bố các lớp sau khi ánh xạ**:
  * `cardboard`: 1,448 ảnh
  * `plastic`: 1,390 ảnh
  * `clothes`: 1,324 ảnh
  * `metal`: 1,581 ảnh
  * `glass`: 1,215 ảnh
  * `paper`: 1,039 ảnh
  * `shoes`: 1,014 ảnh
  * `battery`: 529 ảnh
  * `biological`: 489 ảnh
  * `trash`: 352 ảnh

---

##  Cấu trúc thư mục mã nguồn tiền xử lý

```
src/preprocessing/
├── __init__.py
├── merge_datasets.py       # Gom và chuẩn hóa thư mục raw
├── clean_data.py           # Làm sạch và gán mã hash trùng lặp
├── eda.py                  # Tạo thống kê báo cáo dữ liệu
├── lighting_enhance.py     # CLAHE, gamma, white balance
├── augmentation.py         # Định nghĩa 5 nhóm augmentation (A-E)
├── split_dataset.py        # Chia dữ liệu & áp label mapping
├── balance_classes.py      # Trọng số loss class weights
└── dataloader.py           # Dataset & PyTorch DataLoader
```

## Hướng dẫn chạy nhanh

1. Kích hoạt môi trường ảo `.venv`:
   ```bash
   .venv\Scripts\activate
   ```
2. Chạy toàn bộ pipeline:
   ```bash
   python src/preprocessing/merge_datasets.py
   python src/preprocessing/clean_data.py
   python src/preprocessing/eda.py
   python src/preprocessing/split_dataset.py
   python src/preprocessing/balance_classes.py
   ```
