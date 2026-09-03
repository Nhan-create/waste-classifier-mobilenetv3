# Thiết kế hệ thống phân loại rác MobileNetV3 10 lớp

Ngày: 2026-09-03
Trạng thái: Đã triển khai trên nhánh `feat/mobilenetv3-10-class`.

## 1. Mục tiêu

Mở rộng repository từ một pipeline tiền xử lý thành hệ thống PyTorch hoàn chỉnh có thể:

1. hợp nhất hai bộ dữ liệu Kaggle và chuẩn hóa về đúng 10 lớp;
2. làm sạch, chống rò rỉ dữ liệu và tạo tập train/validation/test có thể tái lập;
3. huấn luyện MobileNetV3-Large bằng transfer learning;
4. đánh giá đa lớp và lưu kết quả có cấu trúc;
5. dự đoán ảnh bằng CLI và ứng dụng PyQt5 hỗ trợ ảnh tĩnh/webcam;
6. chạy toàn bộ quy trình huấn luyện trên Kaggle GPU;
7. lưu checkpoint tự mô tả để thứ tự nhãn không thể bị lệch giữa train và inference.

Không đưa ảnh dataset, Kaggle token hoặc bí mật khác vào Git. Không cố tải checkpoint Hybrid/ViT/ResNet/EfficientNet nhị phân của repository cũ vào MobileNetV3 vì kiến trúc và số đầu ra không tương thích.

## 2. Nguồn dữ liệu và taxonomy

### 2.1 Nguồn

- VN Trash Classification: `mrgetshjtdone/vn-trash-classification`, version 1, giấy phép MIT.
- Garbage Classification v2: `sumn2u/garbage-classification-v2`, version 12, giấy phép MIT.

Các phiên bản được khóa trong cấu hình/notebook để kết quả không thay đổi âm thầm khi chủ dataset phát hành version mới. README phải ghi nguồn, giấy phép và cách trích dẫn.

### 2.2 Mười lớp chuẩn

Thứ tự chuẩn được lưu trong checkpoint và metadata:

1. `battery`
2. `biological`
3. `cardboard`
4. `clothes`
5. `glass`
6. `metal`
7. `paper`
8. `plastic`
9. `shoes`
10. `trash`

Thứ tự này trùng thứ tự alphabet của `torchvision.datasets.ImageFolder`, nhưng mọi consumer vẫn phải đọc `class_names` từ checkpoint thay vì dựa vào giả định alphabet.

### 2.3 Ánh xạ VN Trash

File `data/metadata/label_mapping.csv` được commit và là nguồn sự thật duy nhất:

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

Mười nhãn của Garbage Classification v2 giữ nguyên sau khi chuẩn hóa chữ thường và dấu nối/gạch dưới.

## 3. Pipeline dữ liệu

### 3.1 Tải và hợp nhất

Notebook Kaggle gắn trực tiếp hai dataset làm input. Script local chỉ hỗ trợ tải bằng KaggleHub khi người dùng đã cấu hình xác thực; token không bao giờ được ghi vào file dự án.

`merge_datasets.py` ghi ảnh vào thư mục raw theo nguồn và giữ `source_dataset`, `original_label`, `original_split` trong manifest. Tên đích chứa source và định danh ổn định để không ghi đè file trùng tên.

### 3.2 Làm sạch và chống leakage

Mỗi ảnh được kiểm tra bằng Pillow, chuyển thử sang RGB và ghi:

- kích thước, mode và extension;
- SHA-256 để tìm bản sao byte-identical;
- perceptual hash để tạo nhóm ảnh gần trùng;
- source, nhãn gốc, nhãn chuẩn và trạng thái kiểm duyệt.

Ảnh corrupt hoặc không ánh xạ được bị loại với lý do rõ ràng. Một duplicate cluster chỉ được giữ một đại diện. Nếu một cluster chứa nhiều nhãn chuẩn khác nhau, toàn bộ cluster được đưa vào báo cáo xung đột để kiểm duyệt thay vì tự chọn nhãn. Khoảng cách Hamming của pHash là tham số cấu hình; báo cáo tách exact duplicate và near duplicate, không tuyên bố near-duplicate nếu chỉ so sánh hash bằng nhau.

### 3.3 Chia tập

- Hai nguồn công khai mặc định chia 70% train, 15% validation, 15% test.
- Split stratified theo `source_dataset × unified_label`, seed mặc định 42.
- Các ảnh trong cùng duplicate cluster không được xuất hiện ở nhiều split.
- Mỗi split phải chứa đủ 10 lớp; pipeline dừng với thông báo cụ thể nếu thiếu lớp.
- Output được ghi vào một thư mục dataset version mới. Pipeline không xóa hoặc trộn âm thầm với output cũ.
- Sau split, validator quét lại SHA-256/pHash cluster giữa các split và phải báo zero leakage trước khi cho phép train.

Oversampling vật lý không bật mặc định vì tạo thêm bản sao. Trainer dùng class weights tính chỉ từ tập train.

## 4. Kiến trúc phần mềm

```text
configs/
  preprocessing_config.yaml
  model_config.yaml
data/metadata/
  label_mapping.csv
src/
  preprocessing/            # pipeline hiện có, được củng cố
  data/                      # class discovery, manifests, validation
  models/mobilenetv3.py      # model factory
  training/train.py          # train + resume
  evaluation/evaluate.py     # test metrics + plots
  inference/predict.py       # checkpoint loader + top-k prediction
  ui/                        # style/label presentation thuần, dễ test
app.py                       # PyQt5 image/webcam app
notebooks/
  train_mobilenetv3_kaggle.ipynb
tests/
artifacts/                   # ignored: checkpoints và runtime outputs
outputs/                     # metrics, plots, logs
```

Module model, checkpoint và label schema là code dùng chung. Không sao chép lại định nghĩa kiến trúc trong train/evaluate/predict.

## 5. Mô hình và huấn luyện

### 5.1 Kiến trúc

Model mặc định là `torchvision.models.mobilenet_v3_large` với trọng số ImageNet. Classification layer cuối được thay bằng `Linear(in_features, 10)`. Model trả logits; Softmax chỉ chạy ở evaluation/inference.

MobileNetV3-Small không nằm trong phạm vi bản đầu để tránh tăng ma trận thực nghiệm. Factory dùng stable model ID để có thể thêm biến thể sau này mà không đổi định dạng checkpoint.

### 5.2 Cấu hình mặc định

- input: RGB 224×224, ImageNet mean/std;
- augmentation: nhóm C đã có, chỉ áp dụng cho tập train; validation/test chỉ resize và normalize;
- seed: 42 và bật tùy chọn deterministic;
- batch size: 32, cho phép override khi thiếu VRAM;
- loss: weighted cross-entropy với label smoothing 0.1;
- optimizer: AdamW, weight decay `1e-4`;
- phase 1: freeze feature extractor, train head 5 epoch, LR `1e-3`;
- phase 2: unfreeze toàn bộ, tối đa 25 epoch, LR backbone `1e-4`, LR head `3e-4`;
- cosine learning-rate schedule, AMP khi có CUDA, gradient clipping 1.0;
- early stopping patience 7 theo validation macro-F1.

Class weight tensor phải được tạo theo đúng `class_names` của `ImageFolder`, không dựa vào thứ tự key trong JSON.

## 6. Checkpoint contract

Checkpoint deploy tốt nhất chứa:

```text
format_version
model_name
num_classes
class_names
input_size
normalization
model_state
epoch
metrics
dataset_fingerprint
training_config
```

Checkpoint resume bổ sung optimizer, scheduler và AMP scaler state. Cả `best` và `last` đều dùng một envelope thống nhất. Loader kiểm tra schema, `num_classes`, kích thước head và class order trước khi nạp; mismatch phải fail-fast với lỗi dễ hiểu.

`dataset_fingerprint` là SHA-256 của manifest đã sắp xếp cùng với label mapping và cấu hình split, nhờ đó checkpoint có thể truy về đúng phiên bản dữ liệu.

## 7. Evaluation

Evaluator chỉ đọc test set sau khi checkpoint đã được chọn bằng validation. Nó xuất:

- accuracy;
- macro-F1 và weighted-F1;
- precision, recall, F1 và support từng lớp;
- confusion matrix 10×10, cả raw và normalized;
- one-vs-rest macro AUC khi test có đủ các lớp;
- JSON/CSV metrics và PNG plots dưới `outputs/evaluation/<run-id>/`.

Nếu một lớp thiếu trong test, evaluator vẫn xuất report và ghi AUC là `null` kèm cảnh báo thay vì crash.

## 8. Inference và GUI

Predictor lazy-load đúng một checkpoint, dựng transform từ metadata, dùng `torch.inference_mode()`, rồi trả:

- top-1 class và confidence;
- top-3 classes và probabilities;
- cờ `low_confidence` với ngưỡng mặc định 0.55 có thể cấu hình.

Tên hiển thị tiếng Việt được tách khỏi stable class ID:

| Class ID | Hiển thị |
|---|---|
| `battery` | Pin |
| `biological` | Rác hữu cơ |
| `cardboard` | Bìa carton |
| `clothes` | Quần áo |
| `glass` | Thủy tinh |
| `metal` | Kim loại |
| `paper` | Giấy |
| `plastic` | Nhựa |
| `shoes` | Giày dép |
| `trash` | Rác khác |

GUI PyQt5 kế thừa luồng ảnh/webcam của dự án cũ nhưng không dùng logic màu nhị phân. Webcam truyền frame trong bộ nhớ, không ghi JPEG tạm mỗi 30 ms. Inference chạy ngoài UI thread và có giới hạn tần suất để cửa sổ không bị treo. Ảnh tĩnh có thể ghi history; webcam không ghi mỗi frame.

Ứng dụng không đưa ra chỉ dẫn tái chế mang tính pháp lý vì quy định xử lý pin, thủy tinh, nhựa và rác hữu cơ thay đổi theo địa phương.

## 9. Xử lý lỗi

- Thiếu dataset, mapping, split hoặc checkpoint: dừng sớm và chỉ rõ đường dẫn mong đợi.
- Class set/order khác checkpoint: từ chối chạy thay vì trả nhãn sai.
- Ảnh corrupt/không phải RGB: báo file cụ thể; preprocessing ghi lý do loại.
- CUDA OOM: hướng dẫn giảm batch size; không tự động nuốt lỗi hoặc đổi kết quả.
- Camera không mở được: GUI vẫn dùng được ảnh tĩnh.
- Confidence thấp: hiển thị cảnh báo, không đổi top-1 một cách bí mật.

## 10. Kiểm thử và CI

### Unit tests không cần dataset thật

- label mapping của toàn bộ nhãn VN Trash;
- deterministic stratified/group split và không leakage;
- class-weight ordering;
- checkpoint metadata round-trip và mismatch rejection;
- top-k mapping đúng theo `class_names`;
- GUI presentation cho đủ 10 class ID;
- SQLite history chấp nhận mọi lớp.

### Smoke tests cần PyTorch

- MobileNetV3 nhận `[2, 3, 224, 224]` và trả `[2, 10]`;
- một optimizer step trên synthetic images;
- save/load checkpoint cho prediction giống nhau;
- evaluation trên synthetic ImageFolder 10 lớp tạo report và matrix 10×10.

GitHub Actions chạy lint/syntax, unit tests và smoke tests CPU. Training đầy đủ không chạy trong CI.

## 11. Kaggle notebook

Notebook thực hiện tuần tự:

1. gắn đúng hai dataset/version;
2. chạy merge, clean, mapping, split và leakage validation;
3. in thống kê trước/sau mapping;
4. train hai phase;
5. evaluate test đúng một lần;
6. đóng gói checkpoint deploy, metrics, plots và dataset manifest thành output có thể tải xuống.

Notebook không chứa credential. Khi chạy bên trong Kaggle, input được cung cấp qua cơ chế dataset attachment của Kaggle.

## 12. Tiêu chí nghiệm thu

1. Pipeline tạo đúng 10 thư mục lớp trong cả train/val/test và báo zero cross-split duplicate theo validator.
2. Model output có shape `[batch, 10]`; mọi label inference đến từ checkpoint metadata.
3. Train có thể resume và best checkpoint được chọn bằng validation macro-F1.
4. Evaluator tạo đầy đủ JSON/CSV/plots đa lớp mà không dùng logic AUC nhị phân.
5. CLI dự đoán trả top-1/top-3; GUI hiển thị đủ 10 nhãn và không block event loop.
6. Tests/CI pass và README mô tả đúng dữ liệu, lệnh chạy, artifacts và giới hạn.
7. Không có dataset, checkpoint lớn hoặc secret bị commit ngoài chủ đích.

## 13. Phân phối lên GitHub

Thực hiện trên branch `feat/mobilenetv3-10-class`. Không force-push. Sau khi verification đầy đủ, push branch lên `Nhan-create/waste-classifier-mobilenetv3`; chỉ cập nhật `main` bằng fast-forward/merge bình thường khi trạng thái kiểm thử đã được báo rõ.

Máy hiện tại không đủ dung lượng để tải đồng thời hai archive và chưa có runtime PyTorch/GPU. Vì vậy mã nguồn, tests và notebook Kaggle có thể hoàn thiện tại đây; checkpoint đã huấn luyện chỉ được tạo sau một Kaggle GPU run có xác thực của người dùng.
