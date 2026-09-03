# Streamlit Camera and Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested Streamlit frontend that classifies uploaded images, camera snapshots, uploaded videos, and live browser-camera video through the existing ten-class MobileNetV3 checkpoint contract.

**Architecture:** A thin `streamlit_app.py` composes pure/testable web modules. A locked inference service serializes access to the shared lazy predictor; uploaded videos use bounded timestamp sampling and probability smoothing, while a function-based WebRTC callback throttles inference and overlays the last stable result without calling Streamlit APIs from its worker thread.

**Tech Stack:** Python 3.10+, Streamlit 1.51+, streamlit-webrtc 0.76.2, PyAV, OpenCV headless, Pillow, PyTorch, pytest, Streamlit AppTest

**Spec:** `docs/superpowers/specs/2026-09-03-streamlit-camera-video-design.md`

> **Execution override (2026-09-03):** At the user's explicit request, implementation proceeds without adding or running the test steps below. Only dependency, syntax, lint, and repository-safety checks are performed before delivery.

## Global Constraints

- Preserve canonical class order: `battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash`.
- Reuse `WastePredictor`; do not duplicate model architecture, checkpoint loading, transforms, or label mapping.
- The product is full-frame single-label classification, not object detection; never claim bounding boxes or multiple-object detection.
- Default confidence threshold is `0.55`, video/live sampling is `2.0 FPS`, smoothing window is `5`, and uploaded-video analysis is bounded at `300` sampled frames.
- Uploaded images, camera images, uploaded videos, and webcam frames remain in memory and are not inserted into SQLite history.
- WebRTC uses a function-based callback, calls no `st.*` API from that callback, disables audio, and protects shared state with locks.
- A missing checkpoint renders instructions and stops model/media initialization without crashing the Streamlit session.
- Do not commit a model, media, Streamlit secrets, credentials, or generated runtime artifacts.
- Keep the PyQt app working and retain all existing tests.
- Work on `feat/mobilenetv3-10-class`; push only by fast-forward and never force-push.

---

### Task 1: Streamlit dependencies and validated runtime settings

**Files:**
- Create: `src/web/__init__.py`
- Create: `src/web/settings.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: `tests/web/test_settings.py`
- Test: `tests/web/test_web_dependencies.py`

**Interfaces:**
- Consumes: CLI arguments after Streamlit's `--` separator and optional `WASTE_CHECKPOINT` environment variable.
- Produces: `WebSettings`, `parse_web_settings(argv, environ)`, and `checkpoint_problem(settings)`.

- [ ] **Step 1: Write failing dependency and settings tests**

```python
def test_required_web_dependencies_are_declared():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "streamlit>=1.51,<2" in requirements
    assert "streamlit-webrtc>=0.76.2,<0.77" in requirements
    assert "av>=14,<17" in requirements


def test_cli_checkpoint_overrides_environment(tmp_path):
    cli_checkpoint = tmp_path / "cli.pt"
    cli_checkpoint.touch()
    settings = parse_web_settings(
        ["--checkpoint", str(cli_checkpoint), "--video-sample-fps", "3"],
        {"WASTE_CHECKPOINT": "environment.pt"},
    )
    assert settings.checkpoint_path == cli_checkpoint
    assert settings.video_sample_fps == 3.0
    assert checkpoint_problem(settings) is None


def test_missing_checkpoint_is_a_renderable_problem():
    settings = parse_web_settings([], {})
    assert "best.pt" in checkpoint_problem(settings)
```

Add parameterized tests rejecting thresholds outside `[0, 1]`, FPS outside `(0, 10]`, non-positive smoothing windows, and non-positive max frame counts.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/web/test_settings.py tests/web/test_web_dependencies.py -v`

Expected: collection fails because `src.web.settings` is missing and requirement assertions fail.

- [ ] **Step 3: Implement settings and dependency declarations**

```python
@dataclass(frozen=True)
class WebSettings:
    checkpoint_path: Path | None
    device: str = "auto"
    confidence_threshold: float = 0.55
    video_sample_fps: float = 2.0
    max_video_frames: int = 300
    live_inference_fps: float = 2.0
    smoothing_window: int = 5


def parse_web_settings(
    argv: Sequence[str], environ: Mapping[str, str]
) -> WebSettings:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--video-sample-fps", type=float, default=2.0)
    parser.add_argument("--max-video-frames", type=int, default=300)
    parser.add_argument("--live-inference-fps", type=float, default=2.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    values, _ = parser.parse_known_args(list(argv))
    checkpoint = values.checkpoint or (
        Path(environ["WASTE_CHECKPOINT"]) if environ.get("WASTE_CHECKPOINT") else None
    )
    settings = WebSettings(
        checkpoint_path=checkpoint,
        device=values.device,
        confidence_threshold=values.confidence_threshold,
        video_sample_fps=values.video_sample_fps,
        max_video_frames=values.max_video_frames,
        live_inference_fps=values.live_inference_fps,
        smoothing_window=values.smoothing_window,
    )
    validate_settings(settings)
    return settings
```

`checkpoint_problem` returns a Vietnamese message when the path is absent or not a file. Add the three dependency constraints exactly and ignore `.streamlit/secrets.toml` plus `.streamlit/credentials.toml`.

- [ ] **Step 4: Install declared dependencies in the project environment**

Run: `.venv\Scripts\python.exe -m pip install "streamlit>=1.51,<2" "streamlit-webrtc>=0.76.2,<0.77" "av>=14,<17"`

Expected: installation succeeds, then `.venv\Scripts\python.exe -m pip check` reports no broken requirements.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/web/test_settings.py tests/web/test_web_dependencies.py -v`

Expected: all tests pass.

```powershell
git add src/web/__init__.py src/web/settings.py requirements.txt .gitignore tests/web/test_settings.py tests/web/test_web_dependencies.py
git commit -m "feat(web): add Streamlit runtime settings"
```

### Task 2: Thread-safe inference and probability smoothing

**Files:**
- Create: `src/web/inference_service.py`
- Create: `src/web/smoothing.py`
- Test: `tests/web/test_inference_service.py`
- Test: `tests/web/test_smoothing.py`

**Interfaces:**
- Consumes: any predictor exposing `predict_pil(image, top_k=N) -> Prediction` and `Prediction`/`ScoredClass` from `src.inference.predict`.
- Produces: `InferenceService.predict(image, top_k)` and `PredictionSmoother.add(prediction, top_k=3)`.

- [ ] **Step 1: Write failing serialization test**

```python
def test_inference_service_serializes_predictor_calls():
    predictor = ConcurrentProbePredictor()
    service = InferenceService(predictor)
    image = Image.new("RGB", (8, 8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: service.predict(image, top_k=3), range(8)))
    assert len(results) == 8
    assert predictor.maximum_active_calls == 1
```

The probe increments/decrements an active counter under its own lock and briefly waits so concurrent entry would be observable.

- [ ] **Step 2: Write failing smoothing tests**

```python
def test_smoother_averages_the_bounded_window_and_returns_top_three():
    smoother = PredictionSmoother(window_size=2, confidence_threshold=0.55)
    smoother.add(ten_class_prediction(glass=0.8, plastic=0.1), top_k=3)
    result = smoother.add(ten_class_prediction(glass=0.2, plastic=0.7), top_k=3)
    assert [row.class_id for row in result.topk] == ["glass", "plastic", "battery"]
    assert result.top1.probability == pytest.approx(0.5)
    assert result.low_confidence
```

Also verify eviction after `window_size`, rejection when a sample omits one of the ten classes, rejection when index/class mappings change, and `reset()` empties state.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/web/test_inference_service.py tests/web/test_smoothing.py -v`

Expected: missing modules cause collection errors.

- [ ] **Step 4: Implement the locked service and smoother**

```python
class InferenceService:
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._lock = threading.RLock()

    def predict(self, image: Image.Image, *, top_k: int) -> Prediction:
        with self._lock:
            return self.predictor.predict_pil(image, top_k=top_k)
```

`PredictionSmoother` stores immutable `{class_id: (index, probability)}` vectors in a bounded deque. The first sample establishes the exact class/index map; every later sample must match it. It averages each class, sorts by `(-probability, index)`, returns the requested number of `ScoredClass` rows, and recomputes `low_confidence`.

- [ ] **Step 5: Verify GREEN, lint, and commit**

Run: `pytest tests/web/test_inference_service.py tests/web/test_smoothing.py -v`

Run: `ruff check src/web tests/web`

Expected: all pass with no lint findings.

```powershell
git add src/web/inference_service.py src/web/smoothing.py tests/web/test_inference_service.py tests/web/test_smoothing.py
git commit -m "feat(web): add locked inference and temporal smoothing"
```

### Task 3: Bounded uploaded-video analysis

**Files:**
- Create: `src/web/video.py`
- Test: `tests/web/test_video.py`

**Interfaces:**
- Consumes: file-like video bytes, `InferenceService`, `PredictionSmoother`, and PyAV frames.
- Produces: `SamplingPolicy`, `FrameSampler`, `VideoSample`, `VideoAnalysis`, `VideoAnalysisError`, and the `analyze_uploaded_video` entry point.

- [ ] **Step 1: Write failing timestamp sampler tests**

```python
def test_sampler_selects_first_frame_then_respects_target_interval():
    sampler = FrameSampler(SamplingPolicy(sample_fps=2.0, max_frames=3))
    assert sampler.accept(0.0)
    assert not sampler.accept(0.20)
    assert sampler.accept(0.50)
    assert not sampler.accept(0.75)
    assert sampler.accept(1.00)
    assert sampler.limit_reached
```

Add tests for non-monotonic timestamps and policy validation.

- [ ] **Step 2: Write failing bounded-analysis tests using a fake container**

```python
def test_video_analysis_samples_predicts_reports_and_closes_container():
    container = FakeContainer(frames_at=[0.0, 0.2, 0.5, 1.0, 1.5])
    events = []
    result = analyze_uploaded_video(
        io.BytesIO(b"video"), service,
        SamplingPolicy(sample_fps=2.0, max_frames=2),
        smoothing_window=2,
        open_container=lambda _: container,
        on_sample=lambda timestamp, frame, prediction: events.append(timestamp),
    )
    assert [sample.timestamp_seconds for sample in result.samples] == [0.0, 0.5]
    assert result.truncated
    assert events == [0.0, 0.5]
    assert container.closed
    assert service.top_k_calls == [10, 10]
```

Add cases for no video stream, a decode exception that still closes the container, missing `frame.time` fallback, and empty video.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/web/test_video.py -v`

Expected: collection fails because `src.web.video` is missing.

- [ ] **Step 4: Implement video analysis**

```python
@dataclass(frozen=True)
class SamplingPolicy:
    sample_fps: float = 2.0
    max_frames: int = 300
    fallback_fps: float = 30.0


@dataclass(frozen=True)
class VideoSample:
    timestamp_seconds: float
    prediction: Prediction


@dataclass(frozen=True)
class VideoAnalysis:
    samples: tuple[VideoSample, ...]
    truncated: bool
    top1_counts: dict[str, int]
```

Open with injected `open_container` defaulting to `av.open`, select the first video stream, calculate timestamps from `frame.time` or frame index/rate, convert selected frames with `frame.to_image().convert("RGB")`, call `service.predict(rgb_image, top_k=10)`, smooth to top-3, emit the progress callback, and close in `finally`. Wrap PyAV/decode failures in `VideoAnalysisError` without exposing media bytes.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/web/test_video.py -v`

Expected: all tests pass.

```powershell
git add src/web/video.py tests/web/test_video.py
git commit -m "feat(web): add bounded uploaded-video analysis"
```

### Task 4: Throttled live WebRTC callback

**Files:**
- Create: `src/web/live.py`
- Test: `tests/web/test_live.py`

**Interfaces:**
- Consumes: `av.VideoFrame`, `InferenceService`, `PredictionSmoother`, OpenCV, and an injectable monotonic clock.
- Produces: callable `LiveVideoCallback`, `latest_prediction`, `last_error`, and `reset()`.

- [ ] **Step 1: Write failing throttle and color-conversion test**

```python
def test_live_callback_throttles_inference_and_returns_annotated_frames(monkeypatch):
    times = iter((0.0, 0.1, 0.5))
    service = RecordingService()
    callback = LiveVideoCallback(
        service, inference_fps=2.0, smoothing_window=2,
        confidence_threshold=0.55, clock=lambda: next(times),
    )
    frame = av.VideoFrame.from_ndarray(blue_bgr_frame(), format="bgr24")
    outputs = [callback(frame) for _ in range(3)]
    assert len(service.images) == 2
    assert service.images[0].getpixel((0, 0)) == (0, 0, 255)
    assert all(isinstance(output, av.VideoFrame) for output in outputs)
    assert callback.latest_prediction is not None
```

Monkeypatch `cv2.imwrite` to fail if called. Add tests that an inference exception sets `last_error` but returns a frame, and `reset()` clears smoother/result/error state.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/web/test_live.py -v`

Expected: collection fails because `src.web.live` is missing.

- [ ] **Step 3: Implement callback and ASCII overlay**

```python
class LiveVideoCallback:
    def __init__(self, service, *, inference_fps, smoothing_window,
                 confidence_threshold, clock=time.monotonic):
        self._interval = 1.0 / inference_fps
        self._state_lock = threading.RLock()
        self._next_inference_at = None
        self._smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        now = self._clock()
        if self._inference_is_due(now):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction = self._service.predict(Image.fromarray(rgb), top_k=10)
            self._store(self._smoother.add(prediction, top_k=3))
        annotate_bgr_frame(bgr, self.latest_prediction, self.last_error)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")
```

The implementation holds no Streamlit reference, never queues frames, and overlays stable ASCII class IDs plus percentages so OpenCV font rendering is deterministic.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest tests/web/test_live.py -v`

Expected: all tests pass.

```powershell
git add src/web/live.py tests/web/test_live.py
git commit -m "feat(web): add realtime WebRTC classification callback"
```

### Task 5: Four-mode Streamlit application and health smoke

**Files:**
- Create: `src/web/app_logic.py`
- Create: `streamlit_app.py`
- Test: `tests/web/test_app_logic.py`
- Test: `tests/web/test_streamlit_app.py`
- Test: `tests/web/test_streamlit_server.py`
- Modify: `tests/test_package_imports.py`

**Interfaces:**
- Consumes: all `src.web` services, `WastePredictor`, `present_label`, Streamlit widgets, and `webrtc_streamer`.
- Produces: `classify_image_bytes`, `PredictionView`, `prediction_view`, `render_prediction`, and executable `streamlit_app.py`.

- [ ] **Step 1: Write failing image-byte and view-model tests**

```python
def test_image_bytes_use_rgb_and_render_vietnamese_top_three():
    service = RecordingService(sample_prediction())
    payload = png_bytes(Image.new("L", (12, 8), color=120))
    prediction = classify_image_bytes(payload, service)
    view = prediction_view(prediction)
    assert service.images[0].mode == "RGB"
    assert view.display_name == "Thủy tinh"
    assert [row.display_name for row in view.topk] == ["Thủy tinh", "Nhựa", "Kim loại"]
```

Add a corrupt-image case that raises `MediaInputError` and a low-confidence view that retains the top class plus warning.

- [ ] **Step 2: Write failing Streamlit AppTest coverage**

```python
def test_missing_checkpoint_page_explains_how_to_create_best_pt(monkeypatch):
    monkeypatch.delenv("WASTE_CHECKPOINT", raising=False)
    app = AppTest.from_file("streamlit_app.py", default_timeout=10).run()
    assert not app.exception
    assert any("best.pt" in element.value for element in app.error)


def test_ready_page_has_all_four_input_tabs(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    monkeypatch.setenv("WASTE_CHECKPOINT", str(checkpoint))
    app = AppTest.from_file("streamlit_app.py", default_timeout=10).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Tải ảnh", "Chụp ảnh", "Tải video", "Camera trực tiếp"
    ]
```

The placeholder checkpoint is never loaded because prediction is lazy and no media widget has a value.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/web/test_app_logic.py tests/web/test_streamlit_app.py -v`

Expected: missing modules/app cause collection or file errors.

- [ ] **Step 4: Implement app logic and Streamlit composition**

`classify_image_bytes` opens an in-memory `BytesIO`, forces `load()` and RGB, then calls `service.predict(top_k=3)`. `prediction_view` maps all class IDs through `present_label` and formats probabilities without altering IDs.

`streamlit_app.py`:

```python
st.set_page_config(page_title="Phân loại rác 10 lớp", page_icon="♻️", layout="wide")
settings = parse_web_settings(sys.argv[1:], os.environ)
st.title("♻️ Phân loại rác bằng MobileNetV3")
st.info("Mô hình phân loại toàn bộ khung hình; hãy đặt một vật thể rác chính ở giữa ảnh.")
problem = checkpoint_problem(settings)
if problem:
    st.error(problem)
    st.code("streamlit run streamlit_app.py -- --checkpoint artifacts\\run-001\\best.pt")
    st.stop()

service = cached_service(str(settings.checkpoint_path), settings.device,
                         settings.confidence_threshold)
upload_tab, capture_tab, video_tab, live_tab = st.tabs(
    ["Tải ảnh", "Chụp ảnh", "Tải video", "Camera trực tiếp"]
)
```

Image tabs use distinct widget keys. Video tab shows the uploaded file, starts only on “Phân tích video”, updates a progress/latest-frame placeholder through `on_sample`, then renders counts and last smoothed result. Live tab creates a per-session `LiveVideoCallback` and calls:

```python
webrtc_streamer(
    key="waste-live-camera",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
```

- [ ] **Step 5: Add a real headless server health test**

Use a free localhost port, start `python -m streamlit run streamlit_app.py --server.headless true --server.port <port>` with no checkpoint, poll `http://127.0.0.1:<port>/_stcore/health` for at most 15 seconds, assert the response body contains `ok`, then terminate/wait in `finally`. Capture output and include it in assertion failures; never leave the process running.

- [ ] **Step 6: Verify app and health GREEN, then commit**

Run: `pytest tests/web/test_app_logic.py tests/web/test_streamlit_app.py tests/web/test_streamlit_server.py -v`

Run: `python -m streamlit run streamlit_app.py --server.headless true --server.port 8501`

Expected: automated tests pass; manual process reaches a healthy URL and is stopped without opening a real camera.

```powershell
git add src/web/app_logic.py streamlit_app.py tests/web/test_app_logic.py tests/web/test_streamlit_app.py tests/web/test_streamlit_server.py tests/test_package_imports.py
git commit -m "feat(web): add image camera and video Streamlit app"
```

### Task 6: Documentation, CI, final verification, and branch delivery

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_repository_contract.py`
- Modify: `docs/superpowers/specs/2026-09-03-streamlit-camera-video-design.md`

**Interfaces:**
- Consumes: final commands, supported media modes, deployment constraints, and verification suite.
- Produces: reproducible user instructions and a remote feature branch containing the tested Streamlit product.

- [ ] **Step 1: Write failing documentation/CI assertions**

```python
def test_readme_documents_all_streamlit_media_modes():
    readme = Path("README.md").read_text(encoding="utf-8")
    for text in ("Tải ảnh", "Chụp ảnh", "Tải video", "Camera trực tiếp"):
        assert text in readme
    assert "streamlit run streamlit_app.py -- --checkpoint" in readme
    assert "không phải object detector" in readme
    assert "HTTPS" in readme and "STUN/TURN" in readme


def test_ci_checks_streamlit_entrypoint():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "ruff check src tests app.py streamlit_app.py scripts" in workflow
    assert "python -m compileall src app.py streamlit_app.py scripts" in workflow
```

- [ ] **Step 2: Run documentation tests and verify RED**

Run: `pytest tests/test_repository_contract.py -v`

Expected: assertions for four modes and `streamlit_app.py` quality gates fail.

- [ ] **Step 3: Update README, CI, and spec status**

Add a “Streamlit ảnh/camera/video” section with install/run commands, four modes, no-checkpoint behavior, full-frame limitation, upload limits, privacy, localhost/HTTPS permissions, and STUN/TURN deployment notes. Keep PyQt documentation as an alternate desktop interface. Update both CI commands to include `streamlit_app.py`; CI still performs no training, media upload, or camera access. Change spec status to “Đã triển khai”.

- [ ] **Step 4: Run fresh final verification**

Run each command independently and inspect its exit code/output:

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\ruff.exe check src tests app.py streamlit_app.py scripts
.venv\Scripts\python.exe -m compileall -q src app.py streamlit_app.py scripts
$env:QT_QPA_PLATFORM='offscreen'
$env:MPLBACKEND='Agg'
.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short --branch
```

Expected: no broken requirements, lint/compile exit `0`, all tests pass, no whitespace errors, and only intended Task 6 files remain before commit.

- [ ] **Step 5: Commit delivery metadata**

```powershell
git add README.md .github/workflows/ci.yml tests/test_repository_contract.py docs/superpowers/specs/2026-09-03-streamlit-camera-video-design.md
git commit -m "docs: add Streamlit media usage and CI coverage"
```

- [ ] **Step 6: Re-run completion gate on committed tree**

Invoke `superpowers:verification-before-completion`, rerun pip check, full Ruff, compileall, full pytest, `git diff --check origin/main...HEAD`, and confirm clean status. Inspect `git log --oneline origin/main..HEAD` and ensure no model/media/secret is tracked.

- [ ] **Step 7: Fast-forward push the existing feature branch**

```powershell
git push origin feat/mobilenetv3-10-class
```

Expected: remote advances from `8a9b448`/the design commit through the Streamlit commits. On rejection, inspect remote changes and stop; never force-push.

## Plan Acceptance

- Tasks 1–2 establish validated settings, compatible web dependencies, serialized model access, and deterministic probability smoothing.
- Tasks 3–4 cover bounded uploaded-video sampling plus nonblocking live WebRTC classification without writing media.
- Task 5 exposes all four requested modes and verifies both the Streamlit element tree and a real headless server health endpoint.
- Task 6 documents the classifier limitation and remote deployment requirements, updates CI, runs the complete regression gate, and delivers the branch safely.
