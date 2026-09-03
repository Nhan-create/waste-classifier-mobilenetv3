# Thiết kế ứng dụng Streamlit cho ảnh, camera và video

Ngày: 2026-09-03  
Trạng thái: Đã triển khai theo kiến trúc và luồng chức năng được duyệt.

## 1. Mục tiêu

Bổ sung một frontend Streamlit dùng chung checkpoint và `WastePredictor` hiện có, cho phép:

1. tải file ảnh lên để phân loại;
2. chụp một ảnh bằng camera trình duyệt để phân loại;
3. tải file video và phân tích các khung hình được lấy mẫu;
4. dùng camera trực tiếp qua WebRTC, hiển thị kết quả phân loại trên video;
5. hiển thị top-1, top-3, độ tin cậy và cảnh báo khi confidence dưới `0.55`.

Ứng dụng là bộ phân loại toàn khung hình, không phải object detector. Người dùng cần đưa một vật thể rác chính vào vùng nhìn. Không vẽ bounding box, không nhận diện nhiều vật thể độc lập và không thay đổi taxonomy 10 lớp.

## 2. Phạm vi và phi mục tiêu

### Trong phạm vi

- Streamlit chạy cục bộ bằng một lệnh và có thể triển khai sau HTTPS.
- Ảnh: JPEG, PNG, WebP và BMP nếu Pillow đọc được.
- Video upload: MP4, MOV, AVI, MKV hoặc WebM nếu PyAV/FFmpeg wheel đọc được.
- Camera snapshot: `st.camera_input`.
- Camera realtime: `streamlit-webrtc` với function-based `video_frame_callback`.
- Cùng nhãn tiếng Việt, icon, màu, top-k và checkpoint contract như PyQt.
- Xử lý media trong bộ nhớ; không ghi ảnh camera hay video upload vào repository.
- Giữ frontend PyQt hiện có; Streamlit là cách chạy bổ sung.

### Ngoài phạm vi

- Object detection, segmentation hoặc bounding box.
- Theo dõi nhiều vật thể qua thời gian.
- Huấn luyện model bên trong giao diện Streamlit.
- Tự tải checkpoint không đáng tin cậy từ người dùng. `torch.load` chỉ đọc checkpoint do pipeline của dự án tạo.
- Ghi lịch sử camera/video hoặc lưu media người dùng mặc định.
- Xuất video đã annotate; video upload hiển thị tiến độ, frame gần nhất và bản tóm tắt dự đoán.

## 3. Phương án được chọn

Dùng Streamlit cho giao diện chuẩn và `streamlit-webrtc` cho camera trực tiếp.

- `st.file_uploader` xử lý ảnh và video upload.
- `st.camera_input` xử lý ảnh chụp.
- `webrtc_streamer(..., video_frame_callback=...)` xử lý camera liên tục.
- PyAV giải mã video và chuyển đổi `VideoFrame` trong callback.
- OpenCV chỉ vẽ overlay ASCII an toàn lên frame realtime; tên tiếng Việt đầy đủ xuất hiện trong widget Streamlit.

Function-based callback được chọn vì tài liệu `streamlit-webrtc` hiện hành khuyến nghị API này và dự kiến loại bỏ class-based processor trong major release tương lai. Callback chạy ngoài Streamlit main thread, nên không gọi `st.*` bên trong callback và mọi state chia sẻ đều được khóa.

Hai phương án không chọn:

- Chỉ dùng `st.camera_input`: ít dependency hơn nhưng không đáp ứng camera realtime.
- Tạo custom WebRTC component: linh hoạt hơn nhưng tăng đáng kể phạm vi frontend/JavaScript, deployment và kiểm thử mà không cải thiện bài toán phân loại hiện tại.

## 4. Cấu trúc thành phần

```text
streamlit_app.py
  ├── WebSettings / checkpoint validation
  ├── cached InferenceService
  ├── image upload + camera snapshot
  ├── uploaded-video analysis
  └── WebRTC live callback

src/web/
  ├── inference_service.py  # khóa truy cập WastePredictor dùng chung
  ├── smoothing.py          # trung bình xác suất cửa sổ 5 mẫu
  ├── video.py              # chính sách lấy mẫu + phân tích video upload
  ├── live.py               # callback WebRTC, throttle và overlay
  └── settings.py           # parse --checkpoint/device/threshold
```

`streamlit_app.py` chỉ ghép widget và gọi các thành phần trên. Logic lấy mẫu, smoothing, cấu hình và callback được tách để kiểm thử không cần camera thật.

## 5. Hợp đồng cấu hình và khởi động

Lệnh chuẩn:

```powershell
streamlit run streamlit_app.py -- --checkpoint artifacts\run-001\best.pt
```

Tham số ứng dụng:

- `--checkpoint PATH`: bắt buộc nếu biến `WASTE_CHECKPOINT` không tồn tại;
- `--device {auto,cpu,cuda}`: mặc định `auto`;
- `--confidence-threshold FLOAT`: mặc định `0.55`;
- `--video-sample-fps FLOAT`: mặc định `2.0`, giới hạn `(0, 10]`;
- `--max-video-frames INT`: mặc định `300`, phải dương;
- `--live-inference-fps FLOAT`: mặc định `2.0`, giới hạn `(0, 10]`;
- `--smoothing-window INT`: mặc định `5`, phải dương.

`WebSettings` kiểm tra range và checkpoint path. Nếu thiếu hoặc không tồn tại checkpoint, trang vẫn render tiêu đề cùng hướng dẫn tạo `best.pt`, sau đó dừng trước khi khởi tạo model/WebRTC. Không tạo `QApplication` và không phụ thuộc PyQt trong frontend này.

`InferenceService` được cache bằng `st.cache_resource` theo checkpoint path, device và threshold. Nó chứa một `threading.RLock`; mọi lệnh `predict_pil` đi qua cùng lock để tránh static/video/live callbacks gọi chung model đồng thời.

## 6. Luồng ảnh tĩnh

Hai tab ảnh có cùng data flow:

```text
UploadedFile/camera snapshot bytes
  → Pillow Image.open + load + RGB
  → InferenceService.predict(top_k=3)
  → tên tiếng Việt + confidence + top-3 + warning
```

Ứng dụng bắt `UnidentifiedImageError`/`OSError` và nêu rõ input không đọc được. Media không được ghi xuống đĩa và không thêm vào SQLite history.

## 7. Luồng video upload

`analyze_uploaded_video(file_object, service, policy, on_sample)` dùng `av.open` trực tiếp trên file-like object.

### Chính sách lấy mẫu

- Tần suất mặc định: `2.0` mẫu/giây.
- Mẫu đầu tiên luôn được chọn.
- Mỗi mẫu sau chỉ được chọn khi timestamp lớn hơn hoặc bằng timestamp mẫu trước cộng `1 / sample_fps`.
- Dùng `frame.time` khi có; nếu thiếu, suy ra bằng `frame_index / average_rate`; nếu cả hai thiếu thì dùng thứ tự giải mã với fallback 30 FPS.
- Dừng sau `300` mẫu mặc định và đặt `truncated=True` thay vì xử lý video không giới hạn.
- Mỗi frame được chuyển RGB/Pillow rồi gọi `predict(top_k=10)` để smoothing có đủ xác suất 10 lớp.

`on_sample` nhận timestamp, RGB frame và kết quả đã làm mượt để Streamlit cập nhật progress/frame gần nhất. Kết quả cuối chứa timeline nhẹ, không giữ toàn bộ frame trong RAM.

### Làm mượt

`PredictionSmoother(window_size=5, confidence_threshold=0.55)` giữ tối đa năm vector xác suất gần nhất, tính trung bình theo class ID, sắp giảm dần, rồi trả top-3. Index/class ID lấy từ prediction gốc; class set không khớp giữa các mẫu là lỗi thay vì trộn sai nhãn. `low_confidence` được tính lại từ xác suất top-1 đã làm mượt.

Bản tóm tắt video hiển thị:

- lớp gần nhất đã làm mượt;
- top-3;
- số frame đã phân tích;
- thời điểm đầu/cuối;
- phân bố số lần mỗi lớp đứng top-1;
- cảnh báo nếu đã chạm giới hạn mẫu.

## 8. Luồng camera WebRTC

`LiveVideoCallback` là callable truyền cho `video_frame_callback`:

1. nhận `av.VideoFrame`, chuyển `bgr24`;
2. dùng monotonic clock quyết định đã đến kỳ inference hay chưa;
3. nếu đến kỳ, chuyển BGR→RGB→Pillow và gọi service với `top_k=10`;
4. đưa prediction vào smoother và lưu snapshot kết quả mới nhất dưới lock;
5. vẽ nền tối cùng ba dòng `class_id: confidence` bằng OpenCV;
6. trả `av.VideoFrame.from_ndarray(..., format="bgr24")`.

Các frame giữa hai kỳ inference tái sử dụng kết quả gần nhất, vì vậy video vẫn mượt trong khi model chỉ chạy khoảng 2 lần/giây. Callback không gọi Streamlit API, không tạo file và không xếp hàng tác vụ inference. Audio bị tắt.

WebRTC local hoạt động trên `localhost`. Khi triển khai remote, tài liệu phải nêu HTTPS là bắt buộc để trình duyệt cấp camera và có thể cần STUN/TURN để vượt NAT. Cấu hình mặc định dùng Google public STUN; TURN là trách nhiệm môi trường triển khai và không hard-code credential.

## 9. Giao diện

Trang có sidebar trạng thái model và thông số chỉ đọc, cùng bốn tab:

1. **Tải ảnh** — file uploader và kết quả;
2. **Chụp ảnh** — camera snapshot và kết quả;
3. **Tải video** — preview, nút phân tích, progress, latest frame và summary;
4. **Camera trực tiếp** — nút Start/Stop WebRTC và video có overlay.

Kết quả ảnh/video dùng `present_label` để hiển thị icon, tên tiếng Việt và màu. Top-1 không bị thay thế khi confidence thấp; ứng dụng vừa giữ kết quả vừa hiển thị “Kết quả có độ tin cậy thấp; hãy đưa vật thể lại gần, đủ sáng và thử lại.”

Trang luôn nói rõ: “Mô hình phân loại toàn bộ khung hình; hãy đặt một vật thể rác chính ở giữa ảnh.”

## 10. Xử lý lỗi, quyền riêng tư và an toàn

- Checkpoint thiếu/sai contract: hiển thị lỗi và lệnh tạo checkpoint; không chạy camera.
- CUDA được yêu cầu nhưng không có: lỗi rõ ràng từ `WastePredictor`.
- Ảnh corrupt: báo input không đọc được, không làm sập session.
- Video không có video stream, codec lỗi hoặc frame lỗi: báo nguyên nhân thân thiện; container luôn đóng.
- Camera bị từ chối quyền/WebRTC không kết nối: hướng dẫn kiểm tra permission, HTTPS và STUN/TURN.
- Prediction callback lỗi: overlay lỗi ngắn, lưu lỗi gần nhất an toàn dưới lock và tiếp tục trả frame nếu có thể.
- Không log bytes ảnh/video, không lưu media và không đưa tên file người dùng vào exception ngoài phạm vi session.
- Không chấp nhận checkpoint upload từ browser vì checkpoint pickle không đáng tin cậy có thể gây thực thi mã khi load.

## 11. Dependency

Thêm các constraint tương thích đã kiểm tra:

```text
streamlit>=1.51,<2
streamlit-webrtc>=0.76.2,<0.77
av>=14,<17
```

Giữ `opencv-python-headless` để server không cần GUI native. PyQt5 vẫn tồn tại cho app desktop và không được import bởi `streamlit_app.py`.

## 12. Kiểm tra kỹ thuật

Các lệnh kiểm tra kỹ thuật có thể chạy khi cần:

```powershell
pip check
ruff check src tests app.py streamlit_app.py scripts
python -m compileall src app.py streamlit_app.py scripts
$env:QT_QPA_PLATFORM='offscreen'
$env:MPLBACKEND='Agg'
pytest -q
```

CI không truy cập camera thật và không tải media người dùng.

## 13. Tài liệu và phân phối

README được cập nhật với:

- bốn chế độ input;
- lệnh chạy Streamlit;
- yêu cầu `best.pt`;
- giới hạn full-frame classification;
- quyền camera, HTTPS và STUN/TURN;
- định dạng/giới hạn video;
- phân biệt PyQt desktop và Streamlit web.

Thay đổi được commit trên `feat/mobilenetv3-10-class`, chạy verification đầy đủ rồi push fast-forward lên cùng remote branch, không force-push và không merge `main`.

## 14. Tiêu chí nghiệm thu

1. Ảnh upload và camera snapshot trả cùng cấu trúc top-1/top-3 từ một checkpoint.
2. Video upload được lấy mẫu 2 FPS, làm mượt cửa sổ 5 và dừng ở tối đa 300 mẫu.
3. Camera WebRTC hiển thị video liên tục, model chạy tối đa xấp xỉ 2 FPS và không ghi frame ra file.
4. Class order luôn đến từ checkpoint; tên Việt đến từ `present_label`.
5. Thiếu checkpoint/media lỗi/camera lỗi không làm sập Streamlit session.
6. README không gọi classifier là detector và không tuyên bố có checkpoint đã huấn luyện.
7. Dependency được khóa trong `requirements.txt`; CI kiểm tra lint, compile và regression suite hiện có.
