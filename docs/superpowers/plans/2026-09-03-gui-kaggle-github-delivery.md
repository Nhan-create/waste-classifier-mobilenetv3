# GUI, Kaggle, and GitHub Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a responsive Vietnamese ten-class desktop application, a reproducible Kaggle GPU notebook, complete project documentation/CI, and a verified feature branch on the requested GitHub repository.

**Architecture:** Pure presentation and SQLite modules remain independent of PyQt and PyTorch. Static-image and webcam inference use the shared lazy predictor through background Qt threads; the camera worker pulls and processes one in-memory frame at a time, so queued work cannot grow. Kaggle orchestrates the same project CLI rather than duplicating preprocessing, model, or metric code.

**Tech Stack:** Python 3.10+, PyQt5, OpenCV, SQLite, Pillow, NumPy, PyTorch, KaggleHub, Jupyter/nbformat, GitHub Actions, pytest, pytest-qt

**Spec:** `docs/superpowers/specs/2026-09-03-mobilenetv3-10-class-design.md`

## Global Constraints

- Stable class IDs remain `battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash`; UI names are Vietnamese display text only.
- Default low-confidence threshold is `0.55`; the application displays the top prediction and warning together without replacing the class.
- Webcam frames are transferred in memory; no temporary JPEG is written at any interval.
- Model inference never runs on the Qt UI thread, and webcam work is rate-limited with at most one pending frame.
- Static-image predictions may be persisted; webcam predictions are not inserted into history per frame.
- A camera failure is nonfatal and leaves static-image classification usable.
- The application gives no location-dependent legal recycling instructions.
- Kaggle inputs are the pinned dataset versions from the spec and contain no credential in source or notebook output.
- CI performs syntax, unit, GUI headless, and CPU smoke tests; it never downloads full datasets or performs full training.
- Do not commit dataset images, credentials, runtime SQLite databases, model checkpoints, or other large artifacts.
- Work remains on `feat/mobilenetv3-10-class`; no force-push; update `main` only through normal merge/fast-forward after verification is reported.

---

## File Structure

- `src/ui/presentation.py`: pure mapping from stable IDs to Vietnamese names, icons, and colors.
- `src/ui/history.py`: SQLite schema and repository for static-image results.
- `src/ui/workers.py`: one-shot image and bounded webcam inference threads.
- `src/ui/main_window.py`: PyQt widget composition and signal handling.
- `app.py`: minimal application entry point and dependency wiring.
- `scripts/download_datasets.py`: authenticated local KaggleHub helper using pinned handles.
- `notebooks/train_mobilenetv3_kaggle.ipynb`: linear Kaggle GPU workflow using repository CLIs.
- `.github/workflows/ci.yml`: headless CPU verification.

### Task 1: Vietnamese presentation contract and static-image history

**Files:**
- Create: `src/ui/__init__.py`
- Create: `src/ui/presentation.py`
- Create: `src/ui/history.py`
- Test: `tests/ui/test_presentation.py`
- Test: `tests/ui/test_history.py`

**Interfaces:**
- Consumes: `CLASS_NAMES`, `Prediction`, and `ScoredClass` from the earlier plans.
- Produces: `LabelPresentation`, `present_label`, `HistoryEntry`, and `HistoryRepository`.

- [ ] **Step 1: Write failing presentation coverage tests**

```python
from src.data.schema import CLASS_NAMES
from src.ui.presentation import DISPLAY_NAMES_VI, present_label


def test_every_class_has_the_approved_vietnamese_name():
    assert DISPLAY_NAMES_VI == {
        "battery": "Pin", "biological": "Rác hữu cơ", "cardboard": "Bìa carton",
        "clothes": "Quần áo", "glass": "Thủy tinh", "metal": "Kim loại",
        "paper": "Giấy", "plastic": "Nhựa", "shoes": "Giày dép", "trash": "Rác khác",
    }
    assert set(DISPLAY_NAMES_VI) == set(CLASS_NAMES)


def test_unknown_class_uses_neutral_fallback():
    result = present_label("future_class")
    assert result.display_name == "Không xác định"
    assert result.color == "#64748B"
```

- [ ] **Step 2: Run presentation tests and verify failure**

Run: `pytest tests/ui/test_presentation.py -v`

Expected: FAIL because `src.ui.presentation` is missing.

- [ ] **Step 3: Implement pure presentation metadata**

```python
@dataclass(frozen=True)
class LabelPresentation:
    display_name: str
    icon: str
    color: str


DISPLAY_NAMES_VI = {
    "battery": "Pin", "biological": "Rác hữu cơ", "cardboard": "Bìa carton",
    "clothes": "Quần áo", "glass": "Thủy tinh", "metal": "Kim loại",
    "paper": "Giấy", "plastic": "Nhựa", "shoes": "Giày dép", "trash": "Rác khác",
}


def present_label(class_id: str) -> LabelPresentation:
    if class_id not in DISPLAY_NAMES_VI:
        return LabelPresentation("Không xác định", "?", "#64748B")
    icon, color = STYLE_BY_CLASS[class_id]
    return LabelPresentation(DISPLAY_NAMES_VI[class_id], icon, color)
```

`STYLE_BY_CLASS` contains one short Unicode symbol and one accessible hex color for each exact class ID. Its text contains no disposal/legal instruction.

- [ ] **Step 4: Write failing SQLite round-trip tests**

```python
def test_history_accepts_predictions_from_multiple_classes(tmp_path):
    repository = HistoryRepository(tmp_path / "history.sqlite3")
    for index, class_id in enumerate(("glass", "plastic", "battery")):
        repository.add_image_prediction(
            image_path=f"image-{index}.jpg", model_name="mobilenet_v3_large",
            class_id=class_id, confidence=0.6 + index / 10,
            topk_json='[{"class_id":"glass","probability":0.6}]', low_confidence=False,
        )
    rows = repository.list_recent(limit=10)
    assert [row.class_id for row in rows] == ["battery", "plastic", "glass"]
```

Run: `pytest tests/ui/test_history.py -v`

Expected: FAIL because `src.ui.history` is missing.

- [ ] **Step 5: Implement SQLite storage with explicit schema**

```python
CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    image_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    class_id TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    topk_json TEXT NOT NULL,
    low_confidence INTEGER NOT NULL CHECK(low_confidence IN (0, 1))
)
"""
```

`HistoryRepository` opens a new short-lived connection per operation, enables WAL, creates the table in `__init__`, inserts only via `add_image_prediction`, validates the class ID against `CLASS_NAMES`, and orders `list_recent` by descending integer ID. It never exposes an API named `add_webcam_prediction`.

- [ ] **Step 6: Verify and commit presentation/history**

Run: `pytest tests/ui/test_presentation.py tests/ui/test_history.py -v`

Expected: PASS.

```powershell
git add src/ui/__init__.py src/ui/presentation.py src/ui/history.py tests/ui/test_presentation.py tests/ui/test_history.py
git commit -m "feat(ui): add ten-class presentation and image history"
```

### Task 2: Nonblocking static-image and in-memory webcam workers

**Files:**
- Create: `src/ui/workers.py`
- Test: `tests/ui/test_workers.py`

**Interfaces:**
- Consumes: a `WastePredictor` object exposing `predict_pil(image, top_k=3) -> Prediction`.
- Produces: `FramePrediction`, `ImageInferenceWorker`, and `CameraInferenceWorker` Qt threads.

- [ ] **Step 1: Write failing static-image thread test**

```python
def test_static_inference_runs_on_worker_thread(qtbot, fake_predictor, image_path):
    ui_thread = QApplication.instance().thread()
    worker = ImageInferenceWorker(fake_predictor, image_path)
    with qtbot.waitSignal(worker.result_ready, timeout=1000) as signal:
        worker.start()
    worker.wait(1000)
    assert signal.args[0].source_path == image_path
    assert fake_predictor.qt_threads == [worker]
    assert fake_predictor.qt_threads[0] is not ui_thread
```

- [ ] **Step 2: Write failing webcam no-file/backpressure test**

```python
def test_camera_worker_keeps_frames_in_memory(monkeypatch, qtbot, fake_capture, fake_predictor):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: pytest.fail("webcam wrote a temporary file"))
    worker = CameraInferenceWorker(
        fake_predictor, capture_factory=lambda _: fake_capture,
        camera_index=0, interval_ms=100,
    )
    with qtbot.waitSignal(worker.result_ready, timeout=1500):
        worker.start()
    worker.stop()
    worker.wait(1500)
    assert fake_capture.released
    assert fake_predictor.received_images[0].mode == "RGB"
```

- [ ] **Step 3: Run worker tests and verify failure**

Run: `pytest tests/ui/test_workers.py -v`

Expected: FAIL because `src.ui.workers` is missing.

- [ ] **Step 4: Implement one-shot and bounded camera QThreads**

```python
@dataclass(frozen=True)
class FramePrediction:
    rgb_frame: np.ndarray
    prediction: Prediction
    source_path: str | None


class ImageInferenceWorker(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, predictor: WastePredictor, image_path: Path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path

    def run(self) -> None:
        try:
            with Image.open(self.image_path) as image:
                rgb = image.convert("RGB")
                prediction = self.predictor.predict_pil(rgb, top_k=3)
                self.result_ready.emit(FramePrediction(np.asarray(rgb).copy(), prediction, str(self.image_path)))
        except Exception as error:
            self.error.emit(f"Không thể phân loại {self.image_path}: {error}")
```

`CameraInferenceWorker` owns `cv2.VideoCapture` inside `run`, converts BGR frames with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, wraps them with `Image.fromarray`, calls the shared predictor, emits a copied RGB frame and prediction, and waits the remaining portion of `interval_ms` with `self.msleep`. A thread-safe stop event limits work to one current frame; because capture and prediction occur sequentially inside one worker, queued frames cannot accumulate. It releases capture in `finally`; failure to open emits one Vietnamese error and returns.

- [ ] **Step 5: Verify shutdown and nonfatal camera errors**

Add tests asserting `stop()` completes within `1.5` seconds, `release()` is called exactly once, and a capture with `isOpened() == False` emits `error` without raising in the test process.

Run: `pytest tests/ui/test_workers.py -v`

Expected: PASS and no file appears under the test working directory.

- [ ] **Step 6: Commit background workers**

```powershell
git add src/ui/workers.py tests/ui/test_workers.py
git commit -m "feat(ui): add nonblocking image and webcam inference"
```

### Task 3: PyQt5 desktop application composition

**Files:**
- Create: `src/ui/main_window.py`
- Create: `app.py`
- Test: `tests/ui/test_main_window.py`

**Interfaces:**
- Consumes: `WastePredictor`, `Prediction`, `present_label`, `HistoryRepository`, and the worker classes.
- Produces: `WasteClassifierWindow` and executable `python app.py --checkpoint <path>`.

- [ ] **Step 1: Write failing image result and low-confidence UI tests**

```python
def test_window_renders_top_one_top_three_and_warning(qtbot, window, prediction):
    qtbot.addWidget(window)
    window.handle_result(FramePrediction(rgb_frame(), prediction, "glass.jpg"))
    assert "Thủy tinh" in window.result_label.text()
    assert "62.0%" in window.confidence_label.text()
    assert window.top_three.count() == 3
    prediction = dataclasses.replace(prediction, low_confidence=True)
    window.handle_result(FramePrediction(rgb_frame(), prediction, "glass.jpg"))
    assert window.warning_label.isVisible()
    assert "Thủy tinh" in window.result_label.text()
```

- [ ] **Step 2: Write failing camera-failure/static-image-survival test**

```python
def test_camera_error_does_not_disable_image_button(qtbot, window):
    window.handle_camera_error("Không mở được camera")
    assert window.camera_button.isEnabled()
    assert window.select_image_button.isEnabled()
    assert "Không mở được camera" in window.status_label.text()
```

Run: `pytest tests/ui/test_main_window.py -v`

Expected: FAIL because `WasteClassifierWindow` is missing.

- [ ] **Step 3: Implement the window and explicit state transitions**

The window contains: image preview, “Chọn ảnh”, “Bật/Tắt camera”, stable-ID-backed Vietnamese result, confidence, three-row top list, low-confidence warning, and status text. `select_image()` validates the path and starts one `ImageInferenceWorker`; `start_camera()` starts one `CameraInferenceWorker`; toggling or closing calls `stop()` and `wait(1500)`. Disable only the action currently running. Signal handlers, not worker threads, update widgets.

```python
def handle_result(self, frame_result: FramePrediction) -> None:
    prediction = frame_result.prediction
    presentation = present_label(prediction.top1.class_id)
    self.result_label.setText(presentation.display_name)
    self.result_label.setStyleSheet(f"color: {presentation.color}")
    self.confidence_label.setText(f"Độ tin cậy: {prediction.top1.probability:.1%}")
    self.warning_label.setText("Kết quả có độ tin cậy thấp; hãy thử ảnh rõ hơn.")
    self.warning_label.setVisible(prediction.low_confidence)
    self.top_three.clear()
    for scored in prediction.topk:
        self.top_three.addItem(f"{present_label(scored.class_id).display_name}: {scored.probability:.1%}")
    self.show_rgb_frame(frame_result.rgb_frame)
    if frame_result.source_path is not None:
        self.persist_static_result(frame_result)
```

`persist_static_result` serializes top-k with `json.dumps(..., ensure_ascii=False)` and inserts once. Webcam `FramePrediction.source_path` is `None`, so no history insert occurs.

- [ ] **Step 4: Implement minimal application entry point**

```python
def main() -> int:
    arguments = parse_args()
    application = QApplication(sys.argv)
    predictor = WastePredictor(arguments.checkpoint, device=arguments.device,
                               confidence_threshold=arguments.confidence_threshold)
    history = HistoryRepository(arguments.history_db)
    window = WasteClassifierWindow(predictor, history, camera_index=arguments.camera_index)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
```

Arguments are `--checkpoint` (required), `--device` (`auto` default), `--confidence-threshold` (`0.55` default), `--camera-index` (`0` default), and `--history-db` (`outputs/history.sqlite3` default). Missing checkpoint produces an argparse error naming the path before `QApplication` is created.

- [ ] **Step 5: Verify the headless GUI and commit**

Run: `$env:QT_QPA_PLATFORM='offscreen'; pytest tests/ui/test_main_window.py tests/ui/test_workers.py -v`

Expected: PASS with fake predictor/capture and no real checkpoint or camera.

Run: `python app.py --help`

Expected: exit code `0`.

```powershell
git add src/ui/main_window.py app.py tests/ui/test_main_window.py
git commit -m "feat(ui): add responsive ten-class desktop app"
```

### Task 4: Pinned Kaggle download helper and GPU notebook

**Files:**
- Create: `scripts/download_datasets.py`
- Create: `notebooks/train_mobilenetv3_kaggle.ipynb`
- Test: `tests/scripts/test_download_datasets.py`
- Test: `tests/notebooks/test_kaggle_notebook.py`

**Interfaces:**
- Consumes: pinned source config, data pipeline CLI, training CLI, evaluator CLI, and predictor checkpoint contract.
- Produces: authenticated local dataset roots and one valid linear Kaggle notebook.

- [ ] **Step 1: Write failing download-helper tests without network access**

```python
def test_downloader_uses_exact_pinned_handles(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(kagglehub, "dataset_download", lambda handle, output_dir: calls.append((handle, output_dir)) or str(output_dir))
    result = download_pinned_datasets(tmp_path)
    assert calls == [
        ("mrgetshjtdone/vn-trash-classification/versions/1", tmp_path / "vn_trash"),
        ("sumn2u/garbage-classification-v2/versions/12", tmp_path / "garbage_v2"),
    ]
    assert set(result) == {"vn_trash", "garbage_v2"}
```

- [ ] **Step 2: Run helper tests and verify failure**

Run: `pytest tests/scripts/test_download_datasets.py -v`

Expected: FAIL because `scripts.download_datasets` is missing.

- [ ] **Step 3: Implement an authenticated helper with no secret persistence**

```python
PINNED_DATASETS = {
    "vn_trash": "mrgetshjtdone/vn-trash-classification/versions/1",
    "garbage_v2": "sumn2u/garbage-classification-v2/versions/12",
}


def download_pinned_datasets(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    roots: dict[str, Path] = {}
    for name, handle in PINNED_DATASETS.items():
        destination = output_root / name
        try:
            roots[name] = Path(kagglehub.dataset_download(handle, output_dir=destination))
        except Exception as error:
            raise RuntimeError(
                f"Could not download {handle}; configure KaggleHub authentication outside the repository"
            ) from error
    return roots
```

The helper never reads, prints, or writes a token and has a CLI `--output-root data/sources`.

- [ ] **Step 4: Create and test the notebook contract**

The notebook must contain these executable sections in this exact order: environment/repository setup; locate two Kaggle attached input roots; write resolved paths without credentials; run the project data pipeline; print pre/post mapping statistics and leakage report; train the two phases; evaluate the chosen best checkpoint once on test; package `best.pt`, `metrics.json`, CSV/PNG plots, `split_manifest.csv`, mapping, and resolved configs into `/kaggle/working/waste-classifier-output.zip`.

```python
def test_kaggle_notebook_has_required_order_and_no_credentials():
    notebook = nbformat.read("notebooks/train_mobilenetv3_kaggle.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    markers = ["run_pipeline", "validation.is_valid", "src.training.train", "src.evaluation.evaluate", "waste-classifier-output.zip"]
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "kaggle.json" not in source
    assert "KAGGLE_KEY" not in source
    assert notebook.metadata.kernelspec.name == "python3"
```

Create the notebook using `nbformat.v4.new_notebook`/`new_code_cell` in the implementation patch, save valid notebook JSON, and keep every command based on repository modules rather than copied model or preprocessing code.

- [ ] **Step 5: Verify and commit Kaggle support**

Run: `pytest tests/scripts/test_download_datasets.py tests/notebooks/test_kaggle_notebook.py -v`

Expected: PASS without network access.

```powershell
git add scripts/download_datasets.py notebooks/train_mobilenetv3_kaggle.ipynb tests/scripts/test_download_datasets.py tests/notebooks/test_kaggle_notebook.py
git commit -m "feat(kaggle): add pinned end-to-end GPU workflow"
```

### Task 5: Dependencies, documentation, ignore rules, and CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_package_imports.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `Tienxulidulieu.md`

**Interfaces:**
- Consumes: every CLI and artifact path from all three plans.
- Produces: reproducible setup instructions and a CPU/headless GitHub Actions quality gate.

- [ ] **Step 1: Write the package import smoke test**

```python
@pytest.mark.parametrize("module", [
    "src.data.schema", "src.data.validation", "src.models.mobilenetv3",
    "src.training.train", "src.evaluation.evaluate", "src.inference.predict",
    "src.ui.main_window",
])
def test_public_modules_import(module):
    importlib.import_module(module)
```

Run: `pytest tests/test_package_imports.py -v`

Expected: FAIL until all dependency declarations and package initializers are complete.

- [ ] **Step 2: Pin compatible dependency floors and protect generated files**

`requirements.txt` must include these exact compatible constraints: `torch==2.3.1`, `torchvision==0.18.1`, `Pillow>=10.3,<13`, `ImageHash>=4.3,<5`, `numpy>=1.26,<3`, `pandas>=2.2,<4`, `PyYAML>=6,<7`, `scikit-learn>=1.4,<2`, `matplotlib>=3.8,<4`, `seaborn>=0.13,<1`, `PyQt5>=5.15.10,<6`, `opencv-python-headless==4.10.0.84`, `kagglehub>=1.0.2,<2`, `nbformat>=5.10,<6`, `pytest>=8,<10`, `pytest-qt>=4.4,<5`, and `ruff>=0.5,<1`. Keep runtime and test packages in this one existing file.

`.gitignore` must cover: `data/raw/`, `data/interim/`, `data/processed/`, `data/sources/`, `data/metadata/*/`, `artifacts/`, `outputs/*.sqlite3*`, `*.pt`, `*.pth`, `kaggle.json`, `.kaggle/`, Python caches, pytest cache, notebook checkpoints, and generated archives. Add `!data/metadata/label_mapping.csv` after metadata rules.

- [ ] **Step 3: Write the exact CI workflow**

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      QT_QPA_PLATFORM: offscreen
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: pip install -r requirements.txt
      - run: ruff check src tests app.py scripts
      - run: python -m compileall src app.py scripts
      - run: pytest -q
```

No CI step downloads Kaggle data, accesses a secret, opens a camera, or performs full training.

- [ ] **Step 4: Rewrite the README around the completed workflow**

README sections must be: overview/architecture; exact ten classes and Vietnamese names; both Kaggle source/version/license links; mapping policy; environment install; authenticated local download; attached-input Kaggle workflow; data pipeline commands; training/resume; evaluation; CLI prediction; PyQt image/webcam usage; artifact layout; tests/CI; limitations. State explicitly that the repository does not include data or trained weights and that a trained checkpoint requires a Kaggle GPU run on this machine's constraints.

- [ ] **Step 5: Run repository-wide checks and commit**

Run: `python -m compileall src app.py scripts`

Expected: exit code `0`.

Run: `ruff check src tests app.py scripts`

Expected: exit code `0` with no lint findings.

Run: `$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; pytest -q`

Expected: all tests PASS.

```powershell
git add .github/workflows/ci.yml tests/test_package_imports.py requirements.txt .gitignore README.md Tienxulidulieu.md
git commit -m "docs: add end-to-end usage and CPU CI"
```

### Task 6: Final verification and safe GitHub branch push

**Files:**
- Verify only: all tracked project files.
- Remote destination: `https://github.com/Nhan-create/waste-classifier-mobilenetv3`.

**Interfaces:**
- Consumes: completed commits from every prior task.
- Produces: one pushed `feat/mobilenetv3-10-class` branch with a clean, auditable status.

- [ ] **Step 1: Inspect tracked and ignored material before push**

Run: `git status --short --branch`

Expected: branch `feat/mobilenetv3-10-class`; no unintended modified/untracked files.

Run: `git status --ignored --short`

Expected: datasets, checkpoints, Kaggle credentials, archives, and runtime database paths are ignored; `data/metadata/label_mapping.csv` is tracked.

- [ ] **Step 2: Run all verification commands from a clean environment**

Run: `python -m compileall src app.py scripts`

Expected: exit code `0`.

Run: `ruff check src tests app.py scripts`

Expected: exit code `0` with no lint findings.

Run: `$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; pytest -q`

Expected: all tests PASS with the final count reported.

Run: `git diff --check origin/main...HEAD`

Expected: no whitespace errors.

- [ ] **Step 3: Review commit scope and remote target**

Run: `git remote -v`

Expected: `origin` fetch/push points to `Nhan-create/waste-classifier-mobilenetv3`.

Run: `git log --oneline --decorate origin/main..HEAD`

Expected: design/plan commits followed by small feature commits in data → model/training → evaluation/inference → UI/Kaggle/docs order.

- [ ] **Step 4: Push the feature branch without force**

Run: `git push -u origin feat/mobilenetv3-10-class`

Expected: branch is created or fast-forwarded on GitHub and upstream tracking is configured. If authentication is unavailable, stop after preserving all local commits and report the exact Git credential error; do not change the remote or force-push.

- [ ] **Step 5: Report artifacts and known external requirement**

Report the branch URL, exact verification commands/results, and explicitly distinguish code completion from model training. The trained `best.pt` is created only after the user runs `notebooks/train_mobilenetv3_kaggle.ipynb` with the two pinned Kaggle inputs and GPU enabled. Do not claim a trained checkpoint exists before that run completes.

## Plan Acceptance

- Tasks 1–3 cover all ten Vietnamese labels, top-3/low-confidence presentation, SQLite static history, in-memory webcam inference, a responsive event loop, safe shutdown, and nonfatal camera errors.
- Task 4 covers pinned local KaggleHub acquisition and the complete attached-input Kaggle GPU sequence without embedding credentials or duplicating project logic.
- Task 5 covers dependency reproducibility, ignored large/secret artifacts, full README usage, and headless CPU CI.
- Task 6 proves clean repository state, full tests/syntax, correct GitHub remote, non-force feature-branch delivery, and honest reporting of the external GPU training requirement.
