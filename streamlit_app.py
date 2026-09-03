"""Streamlit frontend for ten-class waste image and video classification."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from src.inference.predict import Prediction, WastePredictor
from src.ui.presentation import present_label
from src.web.app_logic import (
    MediaInputError,
    classify_image_bytes,
    prediction_view,
)
from src.web.inference_service import InferenceService
from src.web.live import LiveVideoCallback
from src.web.settings import WebSettings, checkpoint_problem, parse_web_settings
from src.web.video import (
    SamplingPolicy,
    VideoAnalysis,
    VideoAnalysisError,
    analyze_uploaded_video,
)

st.set_page_config(
    page_title="Phân loại rác 10 lớp",
    page_icon="♻️",
    layout="wide",
)

RUN_COMMAND = (
    "streamlit run streamlit_app.py -- "
    "--checkpoint artifacts\\ecovision\\best.pt"
)
PREPARE_COMMAND = "python -m scripts.import_ecovision_checkpoint"


@st.cache_resource(show_spinner=False)
def cached_service(
    checkpoint_path: str,
    device: str,
    confidence_threshold: float,
) -> InferenceService:
    """Create one lazy predictor per checkpoint/configuration combination."""

    predictor = WastePredictor(
        Path(checkpoint_path),
        device=device,
        confidence_threshold=confidence_threshold,
    )
    return InferenceService(predictor)


def render_prediction(prediction: Prediction) -> None:
    """Render a consistent Vietnamese top-1/top-3 result block."""

    view = prediction_view(prediction)
    st.markdown(f"### {view.icon} {view.display_name}")
    st.metric("Độ tin cậy", f"{view.probability:.1%}")
    if view.low_confidence:
        st.warning(
            "Kết quả có độ tin cậy thấp. Hãy đưa vật thể lại gần, "
            "chụp đủ sáng và thử lại."
        )
    st.markdown("**Top 3 dự đoán**")
    for rank, row in enumerate(view.topk, start=1):
        left, right = st.columns([4, 1])
        left.write(f"{rank}. {row.icon} {row.display_name}")
        right.write(f"{row.probability:.1%}")
        st.progress(min(max(row.probability, 0.0), 1.0))


def render_image_input(payload: bytes, *, caption: str, service: InferenceService) -> None:
    try:
        image, prediction = classify_image_bytes(payload, service)
    except (MediaInputError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return
    preview, result = st.columns([3, 2], gap="large")
    with preview:
        st.image(image, caption=caption, width="stretch")
    with result:
        render_prediction(prediction)


def render_video_summary(analysis: VideoAnalysis) -> None:
    samples = analysis.samples
    last = samples[-1]
    count_col, start_col, end_col = st.columns(3)
    count_col.metric("Khung hình đã phân tích", len(samples))
    start_col.metric("Bắt đầu", f"{samples[0].timestamp_seconds:.1f} giây")
    end_col.metric("Kết thúc", f"{last.timestamp_seconds:.1f} giây")
    if analysis.truncated:
        st.warning("Video đã chạm giới hạn số khung hình phân tích.")
    st.markdown("#### Kết quả gần nhất đã làm mượt")
    render_prediction(last.prediction)
    st.markdown("#### Phân bố top-1 theo khung hình")
    rows = []
    for class_id, count in sorted(
        analysis.top1_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        label = present_label(class_id)
        rows.append(
            {
                "Lớp": f"{label.icon} {label.display_name}",
                "Class ID": class_id,
                "Số khung hình": count,
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_uploaded_video(
    service: InferenceService,
    settings: WebSettings,
) -> None:
    uploaded = st.file_uploader(
        "Chọn video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="uploaded-video",
        help="Video được xử lý trong bộ nhớ và không được lưu vào lịch sử.",
    )
    if uploaded is None:
        st.info("Tải một video có vật thể rác chính để bắt đầu.")
        return

    payload = uploaded.getvalue()
    st.video(payload)
    if not st.button("Phân tích video", type="primary", key="analyze-video"):
        return

    progress = st.progress(0.0, text="Đang chuẩn bị video…")
    latest_frame = st.empty()
    status = st.empty()
    sampled_count = 0

    def on_sample(
        timestamp: float,
        image: Image.Image,
        prediction: Prediction,
    ) -> None:
        nonlocal sampled_count
        sampled_count += 1
        fraction = min(sampled_count / settings.max_video_frames, 1.0)
        progress.progress(
            fraction,
            text=f"Đã phân tích {sampled_count} khung hình…",
        )
        latest_frame.image(
            image,
            caption=f"Khung hình gần nhất: {timestamp:.1f} giây",
            width="stretch",
        )
        latest = prediction_view(prediction)
        status.info(
            f"Gần nhất: {latest.icon} {latest.display_name} "
            f"({latest.probability:.1%})"
        )

    try:
        with st.spinner("Đang giải mã và phân loại video…"):
            analysis = analyze_uploaded_video(
                io.BytesIO(payload),
                service,
                SamplingPolicy(
                    sample_fps=settings.video_sample_fps,
                    max_frames=settings.max_video_frames,
                ),
                smoothing_window=settings.smoothing_window,
                confidence_threshold=settings.confidence_threshold,
                on_sample=on_sample,
            )
    except (VideoAnalysisError, RuntimeError, ValueError) as error:
        progress.empty()
        status.empty()
        st.error(str(error))
        return
    progress.progress(1.0, text="Đã phân tích xong video.")
    render_video_summary(analysis)


def _session_live_callback(
    service: InferenceService,
    settings: WebSettings,
) -> LiveVideoCallback:
    identity = (
        str(settings.checkpoint_path),
        settings.device,
        settings.confidence_threshold,
        settings.live_inference_fps,
        settings.smoothing_window,
    )
    if st.session_state.get("live-callback-identity") != identity:
        st.session_state["live-callback"] = LiveVideoCallback(
            service,
            inference_fps=settings.live_inference_fps,
            smoothing_window=settings.smoothing_window,
            confidence_threshold=settings.confidence_threshold,
        )
        st.session_state["live-callback-identity"] = identity
    return st.session_state["live-callback"]


def render_live_camera(
    service: InferenceService,
    settings: WebSettings,
) -> None:
    st.info(
        "Nhấn START, cho phép trình duyệt dùng camera và đặt một vật thể rác "
        "chính ở giữa khung hình. Kết quả top-3 được vẽ trực tiếp lên video."
    )
    callback = _session_live_callback(service, settings)
    webrtc_streamer(
        key="waste-live-camera",
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )
    if st.button("Xóa kết quả camera", key="reset-live-result"):
        callback.reset()
        st.success("Đã xóa kết quả làm mượt của camera.")
    st.caption(
        "Khi triển khai qua mạng, trang cần HTTPS và môi trường có thể cần "
        "máy chủ TURN để camera kết nối ổn định."
    )


def main() -> None:
    st.title("♻️ Phân loại rác bằng MobileNetV3")
    st.info(
        "Mô hình phân loại toàn bộ khung hình; hãy đặt một vật thể rác "
        "chính ở giữa ảnh. Đây không phải object detector."
    )

    try:
        settings = parse_web_settings(sys.argv[1:], os.environ)
    except (ValueError, SystemExit) as error:
        st.error(f"Cấu hình không hợp lệ: {error}")
        st.code(RUN_COMMAND, language="powershell")
        st.stop()

    problem = checkpoint_problem(settings)
    if problem:
        st.error(problem)
        st.markdown(
            "Tạo checkpoint chạy thử đã xác minh từ model MIT được ghim phiên bản:"
        )
        st.code(PREPARE_COMMAND, language="powershell")
        st.markdown("Sau đó khởi động ứng dụng:")
        st.code(RUN_COMMAND, language="powershell")
        st.stop()

    service = cached_service(
        str(settings.checkpoint_path),
        settings.device,
        settings.confidence_threshold,
    )
    with st.sidebar:
        st.header("Trạng thái")
        st.success("Đã tìm thấy checkpoint")
        st.caption(str(settings.checkpoint_path))
        st.metric("Ngưỡng tin cậy", f"{settings.confidence_threshold:.0%}")
        st.metric("Video / camera", f"{settings.video_sample_fps:g} FPS")
        st.metric("Cửa sổ làm mượt", settings.smoothing_window)
        st.caption("10 lớp: battery, biological, cardboard, clothes, glass, "
                   "metal, paper, plastic, shoes, trash")

    upload_tab, capture_tab, video_tab, live_tab = st.tabs(
        ["Tải ảnh", "Chụp ảnh", "Tải video", "Camera trực tiếp"]
    )
    with upload_tab:
        uploaded_image = st.file_uploader(
            "Chọn ảnh rác",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="uploaded-image",
        )
        if uploaded_image is None:
            st.info("Tải một ảnh JPEG, PNG, WebP hoặc BMP để phân loại.")
        else:
            render_image_input(
                uploaded_image.getvalue(),
                caption="Ảnh đã tải lên",
                service=service,
            )

    with capture_tab:
        captured_image = st.camera_input(
            "Chụp vật thể rác",
            key="camera-snapshot",
        )
        if captured_image is None:
            st.info("Cho phép camera, căn vật thể vào giữa rồi nhấn chụp.")
        else:
            render_image_input(
                captured_image.getvalue(),
                caption="Ảnh vừa chụp",
                service=service,
            )

    with video_tab:
        render_uploaded_video(service, settings)

    with live_tab:
        render_live_camera(service, settings)


if __name__ == "__main__":
    main()
