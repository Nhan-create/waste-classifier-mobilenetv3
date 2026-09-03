# Nguồn checkpoint chạy thử

Checkpoint `artifacts/ecovision/best.pt` được tạo cục bộ bởi
`python -m scripts.import_ecovision_checkpoint`; file model không được commit vào Git.

- Nguồn: `AmadFR/ecovision_mobilenetv3` trên Hugging Face.
- Revision cố định: `7c2daeea3f684058ae8a1c9656c50fc7309fc36c`.
- File nguồn: `ecovision_mobilenetv3_ST.pth` (17,1 MB).
- SHA-256: `0d60ec3944396b9e989a5e082ad848d6820a92e461230d620d7a53e39ccd52a1`.
- Giấy phép do model card công bố: MIT.
- Dataset huấn luyện được model card báo cáo: `sumn2u/garbage-classification-v2`.
- Kiến trúc: TorchVision MobileNetV3-Large, input 224×224.
- Thứ tự lớp được xác nhận từ app nguồn: `battery`, `biological`, `cardboard`,
  `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`.

Script kiểm checksum trước, dùng `torch.load(..., weights_only=True)`, nạp state dict
vào kiến trúc 10 lớp với `strict=True`, rồi mới ghi checkpoint có metadata của dự án.

Đây là checkpoint bên thứ ba phục vụ chạy thử giao diện ngay. Các con số trên model
card chưa được repository này đánh giá độc lập. Muốn báo cáo kết quả đồ án trên hai
dataset đã chọn, hãy chạy pipeline huấn luyện của repository và dùng checkpoint
`artifacts/run-001/best.pt` do chính lần chạy đó tạo.
