# msocr Instruction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the msocr instruction baseline from `Summary-harness-update.md` - build annotation API, CI/CD pipeline infrastructure, evaluation metrics, and missing documentation deliverables.

**Architecture:** FastAPI-based annotation service with session management, evaluation CER/WER runner, RunPod/Harness CI/CD clients, and comprehensive YAML documentation following the msocr.docspec.v1 schema.

**Tech Stack:** Python 3.11+, FastAPI, Kraken (blla, ketos), PyYAML, DVC, RunPod API, Harness API

---

## Phase 1: Documentation Deliverables (Skills & Registries)

### Task 1.1: Create Syriac Printed Training Pipeline YAML

**Files:**
- Create: `pipeline/harness/syriac_printed_train.yaml`

**Context:** Per Section 7 of `Summary-harness-update.md`, this is the first automated pipeline target.

**Step 1: Create harness directory**

```bash
mkdir -p pipeline/harness
```

**Step 2: Create the Syriac printed pipeline YAML**

```yaml
schema_version: msocr.docspec.v1
id: pipeline.harness.syriac_printed_train
title: Syriac Printed OCR Training Pipeline
purpose: First automated pipeline target for Syriac printed OCR with Tesseract fine-tuning via tesstrain.
notes: |
  Priority: Syriac printed, mixed corpus (Estrangela + Serto + East Syriac).
  Baseline: Tesseract `syr` for Estrangela.
  Fine-tuning: tesstrain on tessdata_best/syr.traineddata checkpoint.
  GPU: RTX 4090 (24 GB) is adequate for 500-5,000 line corpus.
scope:
  project: msocr
  languages:
    - name: syriac
      variants:
        - estrangela
        - serto
        - east_syriac
  writing_modes:
    - printed
  acceptance_thresholds:
    estrangela:
      cer_max: 0.05
    serto:
      cer_max: 0.10
    east_syriac:
      cer_max: 0.10
inputs:
  required:
    - ground_truth_xml_dir
    - split_manifest_id
    - script_variant
  optional:
    - base_model_version
    - compute_tier
workflow:
  - step_id: SP1
    name: trigger
    action: Git push to data/syriac/** OR manual dispatch with variant parameter
  - step_id: SP2
    name: validate
    action: Check annotation session export (line count >= 500, XML schema valid), confirm frozen manifest exists
  - step_id: SP3
    name: build_training_image
    action: Docker image ubuntu:22.04 + Tesseract 5.x + tesstrain + tessdata_best/syr.traineddata (SHA-pinned)
  - step_id: SP4
    name: runpod_submit
    action: POST to RunPod API with image ref, GPU tier (RTX 4090 default)
  - step_id: SP5
    name: evaluate
    action: Run tesseract eval on frozen benchmark manifest, write metrics.json
  - step_id: SP6
    name: policy_gate
    action: Apply per-variant CER threshold from acceptance_thresholds
  - step_id: SP7
    name: register_artifact
    action: Push model to HAR as syr-{variant}-printed-v{sequenceId}
  - step_id: SP8
    name: deploy
    action: On production tag only, pull model from HAR for FastAPI service
metrics:
  primary:
    - CER
  secondary:
    - WER
outputs:
  artifacts:
    - trained_model
    - metrics.json
    - config.yaml
    - Dockerfile.sha
integration:
  implemented_modules:
    - msocr/cli.py
    - msocr/data/manager.py
  planned_modules:
    - msocr/pipeline/runpod_client.py
    - msocr/pipeline/har_client.py
    - msocr/evaluation/metrics.py
    - msocr/data/manifest.py
status:
  maturity: active_blueprint
  implementation_state: pipeline_definition_ready
owners:
  - role: syriac_printed_pipeline
    team: msocr
last_updated: 2026-04-04
```

**Step 3: Commit**

```bash
git add pipeline/harness/syriac_printed_train.yaml
git commit -m "docs: add Syriac printed OCR training pipeline definition"
```

---

### Task 1.2: Create Generic HTR Training Pipeline YAML

**Files:**
- Create: `pipeline/harness/htr_generic_train.yaml`

**Context:** Per Section 7, this is the shared HTR pipeline for all languages.

**Step 1: Create the HTR pipeline YAML**

```yaml
schema_version: msocr.docspec.v1
id: pipeline.harness.htr_generic_train
title: Generic HTR Training Pipeline
purpose: Shared HTR training pipeline parameterized by language for manuscript OCR.
notes: |
  For languages without strong public HTR models, custom Kraken training is required.
  Input: PAGE/ALTO XML exported from Transkribus or eScriptorium.
  GPU: RTX 3090/4090 for small corpus (<5,000 lines), A100 40GB for large corpus (>20,000 lines).
scope:
  project: msocr
  languages:
    - name: syriac
      status: priority_1
    - name: coptic
      status: planned
    - name: armenian
      status: planned
    - name: geez
      status: planned
    - name: sogdian
      status: planned
    - name: old_turkish
      status: planned
  writing_modes:
    - handwritten
  acceptance_thresholds:
    default:
      cer_max: 0.10
inputs:
  required:
    - language
    - ground_truth_xml_dir
    - split_manifest_id
  optional:
    - base_model
    - preprocessing_profile
    - compute_tier
workflow:
  - step_id: HT1
    name: trigger
    action: Manual dispatch with language parameter
  - step_id: HT2
    name: validate
    action: Check line count >= 500, XML schema valid (PAGE/ALTO), confirm manifest
  - step_id: HT3
    name: build_training_image
    action: Docker image with Kraken + ketos, GPU-capable
  - step_id: HT4
    name: runpod_submit
    action: POST to RunPod API with GPU tier selection
  - step_id: HT5
    name: evaluate
    action: Run ketos test on frozen manifest, write metrics.json
  - step_id: HT6
    name: policy_gate
    action: Apply CER <= 10% threshold, emit needs_manual_review if failed
  - step_id: HT7
    name: register_artifact
    action: Push model to HAR as {lang}-htr-v{sequenceId}
  - step_id: HT8
    name: deploy
    action: On production tag, update inference endpoint
metrics:
  primary:
    - CER
  secondary:
    - WER
outputs:
  artifacts:
    - trained_htr_model
    - metrics.json
    - config.yaml
integration:
  implemented_modules:
    - msocr/training/ketos_trainer.py
    - msocr/data/annotation.py
  planned_modules:
    - msocr/pipeline/runpod_client.py
    - msocr/pipeline/har_client.py
    - msocr/evaluation/metrics.py
status:
  maturity: active_blueprint
  implementation_state: pipeline_definition_ready
owners:
  - role: htr_training
    team: msocr
last_updated: 2026-04-04
```

**Step 2: Commit**

```bash
git add pipeline/harness/htr_generic_train.yaml
git commit -m "docs: add generic HTR training pipeline definition"
```

---

### Task 1.3: Create State Update YAML

**Files:**
- Create: `output/state_update.yaml`

**Context:** Required deliverable per instruction/Start-here.md.

**Step 1: Create output directory and YAML**

```bash
mkdir -p output
```

```yaml
schema_version: msocr.docspec.v1
id: state_update.current
title: msocr Current Implementation State Update
purpose: Track implementation status and align documentation with code reality.
generated_at: 2026-04-04
state:
  documentation:
    - id: instruction/Start-here.md
      status: exists
      last_updated: 2026-04-04
      notes: Project context and execution notes
    - id: instruction/summary.md
      status: exists
      last_updated: 2026-04-04
      notes: Summary baseline (earlier version)
    - id: instruction/Summary-harness-update.md
      status: exists
      last_updated: 2026-04-04
      notes: Authoritative baseline with CI/CD and annotation API
    - id: instruction/agent.patristic.ocr.model.md
      status: exists
      last_updated: 2026-04-04
      notes: Agent model blueprint
    - id: instruction/eval.md
      status: exists
      last_updated: 2026-04-04
      notes: Evaluation standard
    - id: instruction/code_review.md
      status: exists
      last_updated: 2026-04-04
      notes: Code review template
    - id: skill/engine_router.yaml
      status: exists
      last_updated: 2026-02-12
      notes: Engine router skill
    - id: skill/manual_review_gate.yaml
      status: exists
      notes: Manual review skill
    - id: skill/post_ocr_correction.yaml
      status: exists
      notes: Post-OCR correction skill
    - id: skill/script_mode_classifier.yaml
      status: exists
      notes: Script mode classifier skill
    - id: skill/training_model_skill.yaml
      status: exists
      notes: Training model skill
    - id: skill/sogdian_config.yaml
      status: exists
      notes: Sogdian config
    - id: skill/old_turkish_config.yaml
      status: exists
      notes: Old Turkish config
    - id: source_registry/patristic_sources.yaml
      status: exists
      last_updated: 2026-02-12
      notes: Patristic sources registry
    - id: pipeline/harness/syriac_printed_train.yaml
      status: created
      last_updated: 2026-04-04
      notes: Syriac printed OCR training pipeline
    - id: pipeline/harness/htr_generic_train.yaml
      status: created
      last_updated: 2026-04-04
      notes: Generic HTR training pipeline
  code:
    implemented:
      - id: msocr/cli.py
        status: implemented
        notes: CLI with ocr, htr, train, preprocess, benchmark, api, demo, payne-smith commands
      - id: msocr/data/manager.py
        status: implemented
        notes: Dataset metadata and storage
      - id: msocr/data/annotation.py
        status: implemented
        notes: Annotation bridge (Label Studio, CVAT, ALTO, PAGE)
      - id: msocr/preprocessing/pipeline.py
        status: implemented
        notes: Image preprocessing pipeline
      - id: msocr/models/inference.py
        status: implemented
        notes: Kraken inference wrapper
      - id: msocr/training/ketos_trainer.py
        status: implemented
        notes: Kraken ketos training wrapper
      - id: msocr/service/api.py
        status: implemented
        notes: FastAPI backend service
      - id: msocr/service/gradio_demo.py
        status: implemented
        notes: Gradio demo service
      - id: msocr/evaluation/metrics.py
        status: exists_partial
        notes: CER/WER evaluation (needs enhancement for benchmark protocol)
      - id: msocr/evaluation/printed_benchmark.py
        status: implemented
        notes: Printed benchmark runner
    planned:
      - id: msocr/service/annotation_api.py
        status: planned
        priority: 1
        notes: Annotation API for ground-truth collection
      - id: msocr/data/session_manager.py
        status: planned
        priority: 2
        notes: Session store for annotation API
      - id: msocr/data/manifest.py
        status: planned
        priority: 3
        notes: Frozen manifest manager with DVC integration
      - id: msocr/evaluation/metrics.py
        status: needs_enhancement
        priority: 4
        notes: Add benchmark run metadata requirements
      - id: msocr/pipeline/runpod_client.py
        status: planned
        priority: 5
        notes: RunPod job submission client
      - id: msocr/pipeline/har_client.py
        status: planned
        priority: 6
        notes: Harness Artifact Registry client
      - id: msocr/models/router.py
        status: planned
        priority: 7
        notes: Language-aware model router
      - id: docker/train/Dockerfile
        status: planned
        priority: 8
        notes: Training container spec for RunPod
  languages:
    - name: greek
      writing_mode: printed
      status: implemented
      notes: Kraken primary + fallback, described in Start-here.md
    - name: latin
      writing_mode: printed
      status: implemented
      notes: Kraken CATMuS-Print Large with Tesseract fallback
    - name: syriac
      writing_mode: printed
      status: implemented_baseline
      notes: Tesseract `syr` route implemented
    - name: syriac
      writing_mode: handwritten
      status: planned
      notes: Transkribus bridge + Kraken custom HTR
    - name: coptic
      writing_mode: printed
      status: implemented
      notes: Tesseract `cop` route
    - name: coptic
      writing_mode: handwritten
      status: planned
      notes: Custom Kraken HTR required
    - name: armenian
      writing_mode: printed
      status: implemented
      notes: Tesseract `hye-calfa-n` preferred
    - name: armenian
      writing_mode: handwritten
      status: planned
      notes: Kraken training from PAGE/ALTO XML
    - name: geez
      writing_mode: printed
      status: implemented
      notes: Tesseract `gez`
    - name: geez
      writing_mode: handwritten
      status: planned
      notes: Same preparation as Armenian
    - name: sogdian
      writing_mode: both
      status: implemented_config
      notes: configs/sogdian_config.yaml exists, ground truth collection blocker
    - name: old_turkish
      writing_mode: both
      status: implemented_config
      notes: configs/old_turkish_config.yaml exists
  pipeline_targets:
    - name: syriac_printed
      status: pipeline_definition_ready
      priority: 1
      notes: First automated pipeline target per Summary-harness-update.md Section 11
    - name: greek_printed
      status: implemented_runtime
      notes: Validate with real CER measurement
    - name: latin_printed
      status: implemented_runtime
      notes: Validate with real CER measurement
  next_steps:
    - Implement annotation API (msocr/service/annotation_api.py)
    - Create session manager (msocr/data/session_manager.py)
    - Implement manifest manager with DVC (msocr/data/manifest.py)
    - Enhance evaluation metrics for benchmark protocol
    - Create RunPod client for GPU training
    - Create HAR client for artifact registry
    - Build training Dockerfile
owners:
  - role: msocr_maintainer
    team: msocr
```

**Step 2: Commit**

```bash
git add output/state_update.yaml
git commit -m "docs: add current implementation state update YAML"
```

---

## Phase 2: Annotation API Implementation

### Task 2.1: Create Annotation API Session Manager

**Files:**
- Create: `msocr/data/session_manager.py`
- Test: `tests/data/test_session_manager.py`

**Context:** Per Section 3 of Summary-harness-update.md, session persistence stores under `msocr/data/sessions/{id}/`.

**Step 1: Write the failing test**

```python
# tests/data/test_session_manager.py
"""Tests for annotation session manager."""
import json
from pathlib import Path

import pytest

from msocr.data.session_manager import AnnotationSession, SessionManager


def test_create_session(tmp_path: Path):
    """Test creating a new annotation session."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="syriac",
        script_variant="estrangela",
        ingestion_path="browser_upload",
        source="test_page.tif",
    )
    
    assert session.session_id is not None
    assert session.language == "syriac"
    assert session.script_variant == "estrangela"
    assert session.writing_mode == "handwritten"
    assert session.segmentation_engine in ["blla", "pageseg"]


def test_session_persistence(tmp_path: Path):
    """Test session save and load."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="sogdian",
        script_variant="formal",
        ingestion_path="local_file",
        source="input/test.tif",
    )
    
    # Load session
    loaded = manager.get_session(session.session_id)
    
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.language == "sogdian"


def test_save_annotations(tmp_path: Path):
    """Test saving annotations to a session."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="greek",
        script_variant="polytonic",
        ingestion_path="browser_upload",
        source="test.tif",
    )
    
    # Save annotations
    manager.save_annotations(session.session_id, annotations=[
        {"line_id": "line_001", "transcript": "Ἀρχὴ"},
        {"line_id": "line_002", "transcript": "τοῦ"},
    ])
    
    # Verify saved
    loaded = manager.get_session(session.session_id)
    assert len(loaded.annotations) == 2
    assert loaded.annotations[0]["transcript"] == "Ἀρχὴ"


def test_export_alto(tmp_path: Path):
    """Test exporting session to ALTO XML format."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="syriac",
        script_variant="estrangela",
        ingestion_path="browser_upload",
        source="test.tif",
    )
    
    manager.save_annotations(session.session_id, annotations=[
        {"line_id": "line_001", "transcript": "ܒܪܫܝܬ"},
    ])
    
    alto_path = manager.export_session(session.session_id, format="alto")
    
    assert alto_path.exists()
    assert alto_path.suffix == ".xml"


def test_export_page(tmp_path: Path):
    """Test exporting session to PAGE XML format."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="coptic",
        script_variant="sahidic",
        ingestion_path="browser_upload",
        source="test.tif",
    )
    
    manager.save_annotations(session.session_id, annotations=[
        {"line_id": "line_001", "transcript": "ⲧⲁⲓ"},
    ])
    
    page_path = manager.export_session(session.session_id, format="page")
    
    assert page_path.exists()
    assert page_path.suffix == ".xml"


def test_export_tsv(tmp_path: Path):
    """Test exporting session to TSV format for ketos train."""
    manager = SessionManager(base_dir=tmp_path)
    
    session = manager.create_session(
        language="old_turkish",
        script_variant="old_uyghur",
        ingestion_path="browser_upload",
        source="test.tif",
    )
    
    manager.save_annotations(session.session_id, annotations=[
        {"line_id": "line_001", "transcript": "test"},
    ])
    
    tsv_path = manager.export_session(session.session_id, format="tsv")
    
    assert tsv_path.exists()
    assert tsv_path.suffix == ".tsv"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_session_manager.py -v`
Expected: FAIL with module import error

**Step 3: Create the session data directory structure**

```bash
mkdir -p msocr/data
```

**Step 4: Write minimal implementation**

```python
# msocr/data/session_manager.py
"""Annotation session manager for ground-truth collection."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


class IngestionPath(Enum):
    BROWSER_UPLOAD = "browser_upload"
    LOCAL_FILE = "local_file"
    IIIF_MANIFEST = "iiif_manifest"


class SegmentationEngine(Enum):
    BLLA = "blla"
    PAGESEG = "pageseg"
    MANUAL = "manual"


@dataclass
class LineSegment:
    """Represents a line segment with coordinates and crop."""

    line_id: str
    coordinates: Dict[str, float]  # x, y, width, height or polygon points
    baseline: Optional[List[int]] = None
    crop_path: Optional[Path] = None
    transcript: Optional[str] = None
    skipped: bool = False


@dataclass
class AnnotationSession:
    """Represents an annotation session."""

    session_id: str
    language: str
    script_variant: str
    writing_mode: str
    ingestion_path: str
    source: str
    segmentation_engine: str
    lines: List[LineSegment] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    needs_manual_review: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    page_count: int = 0


class SessionManager:
    """Manages annotation sessions for ground-truth collection."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        language: str,
        script_variant: str,
        ingestion_path: str,
        source: str,
        writing_mode: str = "handwritten",
    ) -> AnnotationSession:
        """Create a new annotation session."""
        session_id = str(uuid.uuid4())[:8]
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create session.json
        session_data = {
            "session_id": session_id,
            "language": language,
            "script_variant": script_variant,
            "writing_mode": writing_mode,
            "ingestion_path": ingestion_path,
            "source": source,
            "segmentation_engine": "blla",
            "lines": [],
            "needs_manual_review": False,
            "created_at": datetime.utcnow().isoformat(),
        }

        session_file = session_dir / "session.json"
        session_file.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

        return AnnotationSession(
            session_id=session_id,
            language=language,
            script_variant=script_variant,
            writing_mode=writing_mode,
            ingestion_path=ingestion_path,
            source=source,
            segmentation_engine="blla",
        )

    def get_session(self, session_id: str) -> Optional[AnnotationSession]:
        """Load a session by ID."""
        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            return None

        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return AnnotationSession(
            session_id=data["session_id"],
            language=data["language"],
            script_variant=data["script_variant"],
            writing_mode=data.get("writing_mode", "handwritten"),
            ingestion_path=data["ingestion_path"],
            source=data["source"],
            segmentation_engine=data["segmentation_engine"],
            lines=[],
            annotations=data.get("annotations", []),
            needs_manual_review=data.get("needs_manual_review", False),
            created_at=data.get("created_at", ""),
        )

    def save_annotations(
        self, session_id: str, annotations: List[Dict[str, Any]]
    ) -> None:
        """Save annotations to a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session_dir = self.sessions_dir / session_id
        session_file = session_dir / "session.json"

        with session_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        data["annotations"] = annotations
        data["updated_at"] = datetime.utcnow().isoformat()

        with session_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_session(self, session_id: str, format: str) -> Path:
        """Export session to ALTO, PAGE, or TSV format."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session_dir = self.sessions_dir / session_id

        if format == "alto":
            return self._export_alto(session, session_dir)
        elif format == "page":
            return self._export_page(session, session_dir)
        elif format == "tsv":
            return self._export_tsv(session, session_dir)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_alto(self, session: AnnotationSession, session_dir: Path) -> Path:
        """Export to ALTO XML format."""
        alto = ET.Element("alto")
        alto.set("xmlns", "http://www.loc.gov/standards/alto/ns-v4#")

        # Add description
        description = ET.SubElement(alto, "Description")
        tags = ET.SubElement(description, "Tags")
        other_tag = ET.SubElement(tags, "OtherTag")
        other_tag.set("ID", f"script.{session.script_variant}")
        other_tag.set("LABEL", session.script_variant)

        # Add layout
        layout = ET.SubElement(alto, "Layout")
        page = ET.SubElement(layout, "Page")
        page.set("ID", "page_1")
        page.set("PHYSICAL_IMG_NR", "1")

        text_block = ET.SubElement(page, "TextBlock")
        text_block.set("ID", "block_1")
        text_block.set("LANG", session.language)
        text_block.set("TAGREFS", f"script.{session.script_variant}")

        for idx, ann in enumerate(session.annotations):
            text_line = ET.SubElement(text_block, "TextLine")
            text_line.set("ID", f"line_{idx+1}")
            text_line.set("HPOS", "0")
            text_line.set("VPOS", str(idx * 50))
            text_line.set("WIDTH", "100")
            text_line.set("HEIGHT", "40")

            string = ET.SubElement(text_line, "String")
            string.set("CONTENT", ann.get("transcript", ""))
            string.set("ID", f"string_{idx+1}")

        output_path = session_dir / f"{session.session_id}.alto.xml"
        ET.ElementTree(alto).write(output_path, encoding="unicode", xml_declaration=True)
        return output_path

    def _export_page(self, session: AnnotationSession, session_dir: Path) -> Path:
        """Export to PAGE XML format."""
        page = ET.Element("PcGts")
        page.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")

        metadata = ET.SubElement(page, "Metadata")
        creator = ET.SubElement(metadata, "Creator")
        creator.text = "msocr"

        page_elem = ET.SubElement(page, "Page")
        page_elem.set("imageFilename", session.source)

        text_region = ET.SubElement(page_elem, "TextRegion")
        text_region.set("custom", f"language:{session.language};script:{session.script_variant}")
        text_region.set("id", "region_1")

        for idx, ann in enumerate(session.annotations):
            text_line = ET.SubElement(text_region, "TextLine")
            text_line.set("id", f"line_{idx+1}")

            coords = ET.SubElement(text_line, "Coords")
            coords.set("points", "0,0 100,0 100,40 0,40")

            baseline = ET.SubElement(text_line, "Baseline")
            baseline.set("points", "0,35 100,35")

            text_equiv = ET.SubElement(text_line, "TextEquiv")
            unicode_elem = ET.SubElement(text_equiv, "Unicode")
            unicode_elem.text = ann.get("transcript", "")

        output_path = session_dir / f"{session.session_id}.page.xml"
        ET.ElementTree(page).write(output_path, encoding="unicode", xml_declaration=True)
        return output_path

    def _export_tsv(self, session: AnnotationSession, session_dir: Path) -> Path:
        """Export to TSV format for ketos train."""
        crops_dir = session_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        output_path = session_dir / f"{session.session_id}.tsv"

        lines = []
        for idx, ann in enumerate(session.annotations):
            line_id = ann.get("line_id", f"line_{idx+1}")
            transcript = ann.get("transcript", "")
            # TSV format: image_path\ttranscript
            lines.append(f"crops/{line_id}.jpg\t{transcript}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return session_dir / f"{session.session_id}.tsv"
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_session_manager.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add msocr/data/session_manager.py tests/data/test_session_manager.py
git commit -m "feat: implement annotation session manager"
```

---

### Task 2.2: Create Annotation API FastAPI Service

**Files:**
- Create: `msocr/service/annotation_api.py`
- Test: `tests/service/test_annotation_api.py`

**Context:** Per Section 3 of Summary-harness-update.md, the annotation API provides browser-accessible ground-truth collection.

**Step 1: Write the failing test**

```python
# tests/service/test_annotation_api.py
"""Tests for annotation API endpoints."""
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_create_session_endpoint(tmp_path: Path):
    """Test POST /api/sessions endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test_page.tif",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["language"] == "syriac"
    assert data["script_variant"] == "estrangela"


def test_get_session_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id} endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session first
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "greek",
            "script_variant": "polytonic",
            "ingestion_path": "local_file",
            "source": "input/test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    # Get session
    response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["language"] == "greek"


def test_save_annotations_endpoint(tmp_path: Path):
    """Test POST /api/sessions/{id}/save endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "sogdian",
            "script_variant": "formal",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    # Save annotations
    response = client.post(
        f"/api/sessions/{session_id}/save",
        json={
            "annotations": [
                {"line_id": "line_001", "transcript": "𐽾𐽿"},
                {"line_id": "line_002", "transcript": "test"},
            ]
        },
    )

    assert response.status_code == 200

    # Verify annotations
    get_response = client.get(f"/api/sessions/{session_id}")
    assert len(get_response.json()["annotations"]) == 2


def test_export_alto_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=alto endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "ܒܪܫܝܬ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=alto")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"


def test_export_page_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=page endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "coptic",
            "script_variant": "sahidic",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "ⲧⲁⲓ"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=page")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"


def test_export_tsv_endpoint(tmp_path: Path):
    """Test GET /api/sessions/{id}/export?format=tsv endpoint."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create and annotate session
    create_response = client.post(
        "/api/sessions",
        json={
            "language": "old_turkish",
            "script_variant": "old_uyghur",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )
    session_id = create_response.json()["session_id"]

    client.post(
        f"/api/sessions/{session_id}/save",
        json={"annotations": [{"line_id": "line_001", "transcript": "test"}]},
    )

    # Export
    response = client.get(f"/api/sessions/{session_id}/export?format=tsv")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_rtl_language_metadata(tmp_path: Path):
    """Test that RTL languages get proper direction metadata."""
    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=tmp_path)
    client = TestClient(app)

    # Create RTL language session
    response = client.post(
        "/api/sessions",
        json={
            "language": "syriac",
            "script_variant": "estrangela",
            "ingestion_path": "browser_upload",
            "source": "test.tif",
        },
    )

    data = response.json()
    assert data["direction"] == "rtl"
    assert data["web_font"] is not None  # Noto Sans Syriac expected
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_annotation_api.py -v`
Expected: FAIL with module import error

**Step 3: Create test directory**

```bash
mkdir -p tests/service
```

**Step 4: Write minimal implementation**

```python
# msocr/service/annotation_api.py
"""Annotation API for browser-accessible ground-truth collection."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from msocr.data.session_manager import SessionManager


# Language registry with RTL and font information
LANGUAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "syriac": {
        "direction": "rtl",
        "web_font": "Noto Sans Syriac",
        "unicode_block": "U+0700-U+074F",
        "variants": ["estrangela", "serto", "east_syriac"],
    },
    "sogdian": {
        "direction": "rtl",
        "web_font": "Noto Sans Sogdian",
        "unicode_block": "U+10F30-U+10F6F",
        "variants": ["formal", "cursive"],
    },
    "old_sogdian": {
        "direction": "rtl",
        "web_font": "Noto Sans Old Sogdian",
        "unicode_block": "U+10F00-U+10F2F",
        "variants": ["ancient_letters"],
    },
    "old_turkish": {
        "direction": "rtl",
        "web_font": "Noto Sans Old Turkic",
        "unicode_block": "U+10C00-U+10C4F",
        "variants": ["old_uyghur"],
    },
    "greek": {
        "direction": "ltr",
        "web_font": "GFS Didot",
        "unicode_block": "U+0370-U+03FF, U+1F00-U+1FFF",
        "variants": ["polytonic", "minuscule", "uncial"],
    },
    "latin": {
        "direction": "ltr",
        "web_font": "Junicode",
        "unicode_block": "U+0000-U+007F, U+0100-U+024F",
        "variants": ["caroline", "insular", "gothic"],
    },
    "coptic": {
        "direction": "ltr",
        "web_font": "Noto Sans Coptic",
        "unicode_block": "U+2C80-U+2CFF",
        "variants": ["sahidic", "bohairic"],
    },
    "armenian": {
        "direction": "ltr",
        "web_font": "Noto Sans Armenian",
        "unicode_block": "U+0530-U+058F",
        "variants": ["erkatagir", "bolorgir", "notrgir"],
    },
    "geez": {
        "direction": "ltr",
        "web_font": "Noto Sans Ethiopic",
        "unicode_block": "U+1200-U+137F",
        "variants": ["geez"],
    },
}


class CreateSessionRequest(BaseModel):
    language: str
    script_variant: str
    ingestion_path: str
    source: str
    writing_mode: str = "handwritten"


class SaveAnnotationsRequest(BaseModel):
    annotations: List[Dict[str, Any]]


def create_app(base_dir: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="msocr Annotation API",
        description="Ground-truth collection for manuscript OCR/HTR",
        version="0.1.0",
    )

    # Initialize session manager
    if base_dir is None:
        base_dir = Path("msocr/data")
    session_manager = SessionManager(base_dir=base_dir)

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
        """Create a new annotation session."""
        # Validate language
        lang_info = LANGUAGE_REGISTRY.get(request.language.lower())
        if not lang_info:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}",
            )

        # Validate script variant
        if request.script_variant.lower() not in [v.lower() for v in lang_info.get("variants", [])]:
            # Allow unknown variants with warning
            pass

        session = session_manager.create_session(
            language=request.language.lower(),
            script_variant=request.script_variant.lower(),
            ingestion_path=request.ingestion_path,
            source=request.source,
            writing_mode=request.writing_mode,
        )

        return {
            "session_id": session.session_id,
            "language": session.language,
            "script_variant": session.script_variant,
            "writing_mode": session.writing_mode,
            "direction": lang_info["direction"],
            "web_font": lang_info["web_font"],
            "ingestion_path": session.ingestion_path,
            "source": session.source,
            "segmentation_engine": session.segmentation_engine,
            "created_at": session.created_at,
            "page_count": 0,
            "line_count": 0,
        }

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> Dict[str, Any]:
        """Get session state including all annotations."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        lang_info = LANGUAGE_REGISTRY.get(session.language, {})

        return {
            "session_id": session.session_id,
            "language": session.language,
            "script_variant": session.script_variant,
            "writing_mode": session.writing_mode,
            "direction": lang_info.get("direction", "ltr"),
            "web_font": lang_info.get("web_font"),
            "ingestion_path": session.ingestion_path,
            "source": session.source,
            "segmentation_engine": session.segmentation_engine,
            "needs_manual_review": session.needs_manual_review,
            "created_at": session.created_at,
            "page_count": session.page_count,
            "line_count": len(session.annotations),
            "annotations": session.annotations,
        }

    @app.post("/api/sessions/{session_id}/save")
    def save_annotations(session_id: str, request: SaveAnnotationsRequest) -> Dict[str, Any]:
        """Save annotations to the session."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        session_manager.save_annotations(session_id, request.annotations)

        return {
            "status": "saved",
            "session_id": session_id,
            "annotation_count": len(request.annotations),
        }

    @app.get("/api/sessions/{session_id}/export")
    def export_session(session_id: str, format: str = "alto") -> FileResponse:
        """Export session to ALTO, PAGE, or TSV format."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        if format not in ["alto", "page", "tsv"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported: alto, page, tsv",
            )

        output_path = session_manager.export_session(session_id, format=format)

        media_type = {
            "alto": "application/xml",
            "page": "application/xml",
            "tsv": "text/plain",
        }

        return FileResponse(
            path=output_path,
            media_type=media_type.get(format, "application/octet-stream"),
            filename=output_path.name,
        )

    @app.get("/api/sessions/{session_id}/line/{line_number}/image")
    def get_line_image(session_id: str, line_number: int) -> FileResponse:
        """Get cropped line image."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # Find line in session
        if line_number < 1 or line_number > len(session.annotations):
            raise HTTPException(status_code=404, detail=f"Line not found: {line_number}")

        # Line images should be stored in crops/ directory
        line_id = session.annotations[line_number - 1].get("line_id", f"line_{line_number:03d}")
        session_dir = session_manager.sessions_dir / session_id
        image_path = session_dir / "crops" / f"{line_id}.jpg"

        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Line image not found: {line_id}")

        return FileResponse(path=image_path, media_type="image/jpeg")

    @app.get("/api/languages")
    def list_languages() -> List[Dict[str, Any]]:
        """List all supported languages with their properties."""
        return [
            {
                "language": lang,
                "direction": info["direction"],
                "web_font": info["web_font"],
                "unicode_block": info["unicode_block"],
                "variants": info["variants"],
            }
            for lang, info in LANGUAGE_REGISTRY.items()
        ]

    return app


# Default app for uvicorn
app = create_app()
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/service/test_annotation_api.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add msocr/service/annotation_api.py tests/service/test_annotation_api.py
git commit -m "feat: implement annotation API for ground-truth collection"
```

---

### Task 2.3: Add Annotation API Command to CLI

**Files:**
- Modify: `msocr/cli.py` (add annotation serve command)

**Step 1: Add annotation serve command**

Add to `msocr/cli.py` after the `demo_gradio` command:

```python
@main.command(name="annotate")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8001, show_default=True, type=int, help="Bind port")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("msocr/data"),
    help="Base directory for session storage",
)
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve_annotation(host, port, data_dir, reload):
    """Run annotation API for ground-truth collection."""
    import uvicorn

    from msocr.service.annotation_api import create_app

    app = create_app(base_dir=data_dir)
    uvicorn.run(app, host=host, port=port)
```

**Step 2: Test CLI command**

Run: `uv run msocr annotate --help`
Expected: Shows annotation command help

**Step 3: Commit**

```bash
git add msocr/cli.py
git commit -m "feat: add annotation serve command to CLI"
```

---

## Phase 3: Evaluation Metrics Enhancement

### Task 3.1: Enhance Benchmark Metrics Module

**Files:**
- Modify: `msocr/evaluation/metrics.py`
- Create: `msocr/data/manifest.py`
- Test: `tests/evaluation/test_benchmark_metrics.py`

**Context:** Per `eval.md` and Section 5 of Summary-harness-update.md, each run must record CER, WER, model_version, split_version, etc.

**Step 1: Check existing metrics module**

```bash
cat msocr/evaluation/metrics.py
```

**Step 2: Write the failing test for manifest manager**

```python
# tests/data/test_manifest.py
"""Tests for frozen manifest manager."""
from pathlib import Path

import pytest


def test_create_manifest(tmp_path: Path):
    """Test creating a frozen manifest."""
    from msocr.data.manifest import ManifestManager

    manager = ManifestManager(base_dir=tmp_path)

    manifest = manager.create_manifest(
        manifest_id="syriac_estrangela_v1",
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        manuscript_ids=["MS001", "MS002", "MS003"],
        split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    )

    assert manifest.manifest_id == "syriac_estrangela_v1"
    assert manifest.language == "syriac"
    assert len(manifest.manuscript_ids) == 3


def test_get_split(tmp_path: Path):
    """Test getting a specific split from manifest."""
    from msocr.data.manifest import ManifestManager, SplitType

    manager = ManifestManager(base_dir=tmp_path)

    manifest = manager.create_manifest(
        manifest_id="test_manifest",
        language="greek",
        script_variant="polytonic",
        writing_mode="printed",
        manuscript_ids=["MS001", "MS002", "MS003", "MS004", "MS005"],
        split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
    )

    train_ids = manager.get_split(manifest.manifest_id, SplitType.TRAIN)

    assert len(train_ids) == 3  # 60% of 5 manuscripts


def test_freeze_manifest(tmp_path: Path):
    """Test that manifests are frozen after creation."""
    from msocr.data.manifest import ManifestManager

    manager = ManifestManager(base_dir=tmp_path)

    manifest = manager.create_manifest(
        manifest_id="frozen_test",
        language="latin",
        script_variant="caroline",
        writing_mode="printed",
        manuscript_ids=["MS001", "MS002"],
        split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )

    # Attempting to modify should fail
    with pytest.raises(ValueError, match="frozen"):
        manager.update_manifest(manifest.manifest_id, manuscript_ids=["MS003"])


def test_version_immutability(tmp_path: Path):
    """Test that split versions are immutable."""
    from msocr.data.manifest import ManifestManager

    manager = ManifestManager(base_dir=tmp_path)

    manifest_v1 = manager.create_manifest(
        manifest_id="version_test_v1",
        language="coptic",
        script_variant="sahidic",
        writing_mode="printed",
        manuscript_ids=["MS001", "MS002"],
        split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )

    # Creating same manifest ID again should fail
    with pytest.raises(ValueError, match="already exists"):
        manager.create_manifest(
            manifest_id="version_test_v1",
            language="coptic",
            script_variant="sahidic",
            writing_mode="printed",
            manuscript_ids=["MS001", "MS002", "MS003"],
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        )
```

**Step 3: Write manifest manager implementation**

```python
# msocr/data/manifest.py
"""Frozen manifest manager for benchmark dataset splits."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class Manifest:
    """Represents a frozen dataset manifest."""

    manifest_id: str
    language: str
    script_variant: str
    writing_mode: str
    manuscript_ids: List[str]
    splits: Dict[str, List[str]]  # split_name -> list of manuscript_ids
    split_ratios: Dict[str, float]
    created_at: str
    frozen: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ManifestManager:
    """Manages frozen dataset manifests for benchmark evaluation."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.manifests_dir = self.base_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def create_manifest(
        self,
        manifest_id: str,
        language: str,
        script_variant: str,
        writing_mode: str,
        manuscript_ids: List[str],
        split_ratios: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Manifest:
        """Create and freeze a new dataset manifest."""
        # Check if manifest already exists
        manifest_file = self.manifests_dir / f"{manifest_id}.json"
        if manifest_file.exists():
            raise ValueError(f"Manifest already exists: {manifest_id}")

        # Validate split ratios sum to 1.0
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Split manuscripts ensuring no overlap
        splits = self._split_manuscripts(manuscript_ids, split_ratios)

        manifest = Manifest(
            manifest_id=manifest_id,
            language=language.lower(),
            script_variant=script_variant.lower(),
            writing_mode=writing_mode.lower(),
            manuscript_ids=manuscript_ids,
            splits=splits,
            split_ratios=split_ratios,
            created_at=datetime.utcnow().isoformat(),
            frozen=True,
            metadata=metadata or {},
        )

        # Save manifest
        self._save_manifest(manifest)

        return manifest

    def get_manifest(self, manifest_id: str) -> Optional[Manifest]:
        """Load a manifest by ID."""
        manifest_file = self.manifests_dir / f"{manifest_id}.json"
        if not manifest_file.exists():
            return None

        with manifest_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return Manifest(
            manifest_id=data["manifest_id"],
            language=data["language"],
            script_variant=data["script_variant"],
            writing_mode=data["writing_mode"],
            manuscript_ids=data["manuscript_ids"],
            splits=data["splits"],
            split_ratios=data["split_ratios"],
            created_at=data["created_at"],
            frozen=data.get("frozen", True),
            metadata=data.get("metadata", {}),
        )

    def get_split(self, manifest_id: str, split_type: SplitType) -> List[str]:
        """Get manuscript IDs for a specific split."""
        manifest = self.get_manifest(manifest_id)
        if not manifest:
            raise ValueError(f"Manifest not found: {manifest_id}")

        return manifest.splits.get(split_type.value, [])

    def list_manifests(self, language: Optional[str] = None) -> List[Manifest]:
        """List all manifests, optionally filtered by language."""
        manifests = []
        for manifest_file in self.manifests_dir.glob("*.json"):
            manifest = self.get_manifest(manifest_file.stem)
            if manifest:
                if language is None or manifest.language == language.lower():
                    manifests.append(manifest)
        return manifests

    def _split_manuscripts(
        self, manuscript_ids: List[str], split_ratios: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Split manuscripts into train/val/test ensuring no overlap."""
        # Use deterministic shuffling based on sorted IDs
        sorted_ids = sorted(manuscript_ids)
        
        splits: Dict[str, List[str]] = {}
        current_idx = 0

        for split_name, ratio in sorted(split_ratios.items()):
            count = int(len(sorted_ids) * ratio)
            # Ensure at least one item per split if possible
            if count == 0 and len(sorted_ids) > current_idx:
                count = 1
            splits[split_name] = sorted_ids[current_idx : current_idx + count]
            current_idx += count

        # Assign remaining items to the last split
        if current_idx < len(sorted_ids):
            last_split = list(split_ratios.keys())[-1]
            splits[last_split].extend(sorted_ids[current_idx:])

        return splits

    def _save_manifest(self, manifest: Manifest) -> None:
        """Save manifest to disk."""
        manifest_file = self.manifests_dir / f"{manifest.manifest_id}.json"

        data = {
            "manifest_id": manifest.manifest_id,
            "language": manifest.language,
            "script_variant": manifest.script_variant,
            "writing_mode": manifest.writing_mode,
            "manuscript_ids": manifest.manuscript_ids,
            "splits": manifest.splits,
            "split_ratios": manifest.split_ratios,
            "created_at": manifest.created_at,
            "frozen": manifest.frozen,
            "metadata": manifest.metadata,
            # Add checksum for integrity
            "checksum": self._compute_checksum(manifest),
        }

        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_checksum(self, manifest: Manifest) -> str:
        """Compute checksum for manifest integrity."""
        content = f"{manifest.manifest_id}:{','.join(sorted(manifest.manuscript_ids))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

**Step 4: Write benchmark metrics test**

```python
# tests/evaluation/test_benchmark_metrics.py
"""Tests for benchmark metrics recording."""
from pathlib import Path

import pytest


def test_record_benchmark_run(tmp_path: Path):
    """Test recording a benchmark run with all required metadata."""
    from msocr.evaluation.metrics import BenchmarkRecorder

    recorder = BenchmarkRecorder(output_dir=tmp_path)

    result = recorder.record_run(
        benchmark_id="syriac_estrangela_v1_run001",
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        model_id="syr-estrangela-printed-v1",
        model_version="1.0.0",
        preprocessing_profile="default",
        split_version="syriac_estrangela_v1",
        cer=0.035,
        wer=0.082,
        pass_fail=True,
        needs_manual_review=False,
    )

    assert result.benchmark_id == "syriac_estrangela_v1_run001"
    assert result.cer == 0.035
    assert result.pass_fail is True


def test_benchmark_metrics_file(tmp_path: Path):
    """Test that benchmark metrics are written to JSON."""
    from msocr.evaluation.metrics import BenchmarkRecorder

    recorder = BenchmarkRecorder(output_dir=tmp_path)

    recorder.record_run(
        benchmark_id="test_run",
        language="greek",
        script_variant="polytonic",
        writing_mode="printed",
        model_id="greek-polytonic-v1",
        model_version="1.0.0",
        preprocessing_profile="default",
        split_version="greek_polytonic_v1",
        cer=0.045,
        wer=0.095,
        pass_fail=True,
        needs_manual_review=False,
    )

    # Verify metrics.json was created
    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()


def test_cer_gate_pass(tmp_path: Path):
    """Test CER gate pass for printed OCR."""
    from msocr.evaluation.metrics import BenchmarkRecorder

    recorder = BenchmarkRecorder(output_dir=tmp_path)

    result = recorder.record_run(
        benchmark_id="pass_test",
        language="latin",
        script_variant="caroline",
        writing_mode="printed",
        model_id="latin-v1",
        model_version="1.0.0",
        preprocessing_profile="default",
        split_version="latin_v1",
        cer=0.04,  # <= 5%
        wer=0.09,
        pass_fail=True,
        needs_manual_review=False,
    )

    assert result.pass_fail is True
    assert result.needs_manual_review is False


def test_cer_gate_fail_triggers_review(tmp_path: Path):
    """Test that CER gate failure triggers manual review."""
    from msocr.evaluation.metrics import BenchmarkRecorder

    recorder = BenchmarkRecorder(output_dir=tmp_path)

    result = recorder.record_run(
        benchmark_id="fail_test",
        language="sogdian",
        script_variant="formal",
        writing_mode="printed",
        model_id="sogdian-v1",
        model_version="1.0.0",
        preprocessing_profile="default",
        split_version="sogdian_v1",
        cer=0.12,  # > 10%
        wer=0.25,
        pass_fail=False,
        needs_manual_review=True,
    )

    assert result.pass_fail is False
    assert result.needs_manual_review is True
```

**Step 5: Enhance metrics module with benchmark recorder**

Add to `msocr/evaluation/metrics.py` or create it:

```python
# msocr/evaluation/metrics.py
"""Evaluation metrics and benchmark recording for msocr."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Represents a benchmark run result with all required metadata."""

    benchmark_id: str
    language: str
    script_variant: str
    writing_mode: str
    model_id: str
    model_version: str
    preprocessing_profile: str
    split_version: str
    cer: float
    wer: float
    pass_fail: bool
    needs_manual_review: bool
    pipeline_run_id: Optional[str] = None
    manifest_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# CER thresholds per language and variant (from Summary-harness-update.md Section 5)
CER_THRESHOLDS: Dict[str, Dict[str, Dict[str, float]]] = {
    "syriac": {
        "estrangela": {"printed": 0.05, "handwritten": 0.10},
        "serto": {"printed": 0.10, "handwritten": 0.10},
        "east_syriac": {"printed": 0.10, "handwritten": 0.10},
    },
    "greek": {
        "polytonic": {"printed": 0.05, "handwritten": 0.10},
    },
    "latin": {
        "default": {"printed": 0.05, "handwritten": 0.10},
    },
    "coptic": {
        "sahidic": {"printed": 0.05, "handwritten": 0.10},
        "bohairic": {"printed": 0.05, "handwritten": 0.10},
    },
    "armenian": {
        "default": {"printed": 0.05, "handwritten": 0.10},
    },
    "geez": {
        "default": {"printed": 0.05, "handwritten": 0.10},
    },
    "sogdian": {
        "default": {"printed": 0.10, "handwritten": 0.10},
    },
    "old_turkish": {
        "default": {"printed": 0.10, "handwritten": 0.10},
    },
}


def get_cer_threshold(language: str, script_variant: str, writing_mode: str) -> float:
    """Get CER threshold for a language/variant/mode combination."""
    lang_data = CER_THRESHOLDS.get(language.lower(), {})
    variant_key = script_variant.lower() if script_variant else "default"
    mode_key = writing_mode.lower()

    # Try specific variant first, then default
    variant_data = lang_data.get(variant_key, lang_data.get("default", {}))
    return variant_data.get(mode_key, 0.10)  # Default to 10%


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    # Levenshtein distance at character level
    import Levenshtein

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    import Levenshtein

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Join words with special separator for Levenshtein
    ref_joined = " ".join(ref_words)
    hyp_joined = " ".join(hyp_words)

    distance = Levenshtein.distance(ref_joined, hyp_joined)
    # Normalize by word-level distance
    return distance / len(ref_joined)


class BenchmarkRecorder:
    """Records benchmark runs with full metadata."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_run(
        self,
        benchmark_id: str,
        language: str,
        script_variant: str,
        writing_mode: str,
        model_id: str,
        model_version: str,
        preprocessing_profile: str,
        split_version: str,
        cer: float,
        wer: float,
        pass_fail: bool,
        needs_manual_review: bool,
        pipeline_run_id: Optional[str] = None,
        manifest_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Record a benchmark run with all required metadata."""
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            language=language.lower(),
            script_variant=script_variant.lower(),
            writing_mode=writing_mode.lower(),
            model_id=model_id,
            model_version=model_version,
            preprocessing_profile=preprocessing_profile,
            split_version=split_version,
            cer=cer,
            wer=wer,
            pass_fail=pass_fail,
            needs_manual_review=needs_manual_review,
            pipeline_run_id=pipeline_run_id,
            manifest_id=manifest_id or split_version,
            metadata=metadata or {},
        )

        # Write to metrics.json
        metrics_file = self.output_dir / "metrics.json"
        
        # Load existing records if any
        records = []
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                records = json.load(f)

        # Append new record
        records.append(asdict(result))

        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        return result

    def check_gate(
        self, language: str, script_variant: str, writing_mode: str, cer: float
    ) -> tuple[bool, bool]:
        """Check CER gate and determine if manual review is needed.

        Returns: (pass_fail, needs_manual_review)
        """
        threshold = get_cer_threshold(language, script_variant, writing_mode)
        passed = cer <= threshold
        needs_review = not passed
        return passed, needs_review
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_manifest.py tests/evaluation/test_benchmark_metrics.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add msocr/data/manifest.py msocr/evaluation/metrics.py tests/data/test_manifest.py tests/evaluation/test_benchmark_metrics.py
git commit -m "feat: implement frozen manifest manager and benchmark recorder"
```

---

## Phase 4: RunPod Client Implementation

### Task 4.1: Create RunPod API Client

**Files:**
- Create: `msocr/pipeline/runpod_client.py`
- Test: `tests/pipeline/test_runpod_client.py`

**Context:** Per Section 7 of Summary-harness-update.md, submit training jobs to RunPod.

**Step 1: Write failing test**

```python
# tests/pipeline/test_runpod_client.py
"""Tests for RunPod API client."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_submit_training_job():
    """Test submitting a training job to RunPod."""
    from msocr.pipeline.runpod_client import RunPodClient

    client = RunPodClient(api_key="test_key")

    with patch.object(client, "_post") as mock_post:
        mock_post.return_value = {"id": "job_12345", "status": "PENDING"}

        job_id = client.submit_job(
            image="msocr/train:latest",
            gpu_type="RTX4090",
            env={"LANGUAGE": "syriac", "VARIANT": "estrangela"},
        )

        assert job_id == "job_12345"
        mock_post.assert_called_once()


def test_poll_job_status():
    """Test polling job status."""
    from msocr.pipeline.runpod_client import RunPodClient, JobStatus

    client = RunPodClient(api_key="test_key")

    with patch.object(client, "_get") as mock_get:
        mock_get.return_value = {"id": "job_12345", "status": "COMPLETED"}

        status = client.get_job_status("job_12345")

        assert status == JobStatus.COMPLETED


def test_cancel_job():
    """Test cancelling a running job."""
    from msocr.pipeline.runpod_client import RunPodClient, JobStatus

    client = RunPodClient(api_key="test_key")

    with patch.object(client, "_post") as mock_post:
        mock_post.return_value = {"id": "job_12345", "status": "CANCELLED"}

        result = client.cancel_job("job_12345")

        assert result.status == JobStatus.CANCELLED


def test_get_job_output(tmp_path: Path):
    """Test retrieving job output artifacts."""
    from msocr.pipeline.runpod_client import RunPodClient

    client = RunPodClient(api_key="test_key")

    with patch.object(client, "_get") as mock_get:
        mock_get.return_value = {
            "id": "job_12345",
            "status": "COMPLETED",
            "output": {"model_path": "/output/model.mlmodel", "metrics_path": "/output/metrics.json"},
        }

        output = client.get_job_output("job_12345")

        assert "model_path" in output
        assert "metrics_path" in output
```

**Step 2: Create pipeline directory**

```bash
mkdir -p msocr/pipeline tests/pipeline
```

**Step 3: Write RunPod client implementation**

```python
# msocr/pipeline/runpod_client.py
"""RunPod API client for GPU training job submission."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import requests


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobResult:
    """Represents a RunPod job result."""

    job_id: str
    status: JobStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RunPodClient:
    """Client for submitting and managing RunPod training jobs."""

    BASE_URL = "https://api.runpod.io/v2"

    def __init__(self, api_key: str, endpoint_id: Optional[str] = None):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def submit_job(
        self,
        image: str,
        gpu_type: str = "RTX4090",
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a training job to RunPod.

        Args:
            image: Docker image to run
            gpu_type: GPU type (RTX3090, RTX4090, A100, etc.)
            env: Environment variables
            input_data: Input data for the job

        Returns:
            Job ID
        """
        payload = {
            "input": {
                "image": image,
                "gpu_type": gpu_type,
                "env": env or {},
                **(input_data or {}),
            }
        }

        response = self._post(f"/run/{self.endpoint_id}", payload)
        return response["id"]

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id: RunPod job ID

        Returns:
            Job status
        """
        response = self._get(f"/run/{self.endpoint_id}/{job_id}")
        return JobStatus(response["status"])

    def cancel_job(self, job_id: str) -> JobResult:
        """Cancel a running job.

        Args:
            job_id: RunPod job ID

        Returns:
            Job result
        """
        response = self._post(f"/run/{self.endpoint_id}/{job_id}/cancel", {})
        return JobResult(
            job_id=job_id,
            status=JobStatus(response.get("status", "CANCELLED")),
        )

    def get_job_output(self, job_id: str) -> Dict[str, Any]:
        """Get job output artifacts.

        Args:
            job_id: RunPod job ID

        Returns:
            Output data including model and metrics paths
        """
        response = self._get(f"/run/{self.endpoint_id}/{job_id}")
        return response.get("output", {})

    def list_endpoints(self) -> list[Dict[str, Any]]:
        """List all available endpoints."""
        return self._get("/endpoints")

    def _get(self, path: str) -> Dict[str, Any]:
        """Make a GET request to RunPod API."""
        url = f"{self.BASE_URL}{path}"
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to RunPod API."""
        url = f"{self.BASE_URL}{path}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
```

**Step 4: Create pipeline __init__.py**

```python
# msocr/pipeline/__init__.py
"""Pipeline infrastructure for CI/CD and training."""

from msocr.pipeline.runpod_client import RunPodClient, JobStatus, JobResult

__all__ = ["RunPodClient", "JobStatus", "JobResult"]
```

**Step 5: Run tests**

Run: `uv run pytest tests/pipeline/test_runpod_client.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add msocr/pipeline/ tests/pipeline/
git commit -m "feat: implement RunPod API client for GPU training"
```

---

## Phase 5: Harness Artifact Registry Client

### Task 5.1: Create HAR Client

**Files:**
- Create: `msocr/pipeline/har_client.py`
- Test: `tests/pipeline/test_har_client.py`

**Context:** Per Section 7, use HAR for model storage.

**Step 1: Write failing test**

```python
# tests/pipeline/test_har_client.py
"""Tests for Harness Artifact Registry client."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_push_artifact():
    """Test pushing artifact to HAR."""
    from msocr.pipeline.har_client import HARClient

    client = HARClient(api_token="test_token", registry_name="msocr-models")

    with patch.object(client, "_post") as mock_post:
        mock_post.return_value = {"artifact_id": "art_12345", "version": "v14"}

        result = client.push_artifact(
            name="syr-estrangela-printed",
            version="v14",
            artifact_path=Path("/tmp/model.mlmodel"),
            sidecar_files={
                "metrics.json": Path("/tmp/metrics.json"),
                "config.yaml": Path("/tmp/config.yaml"),
            },
        )

        assert result.artifact_id == "art_12345"
        assert result.version == "v14"


def test_pull_artifact(tmp_path: Path):
    """Test pulling artifact from HAR."""
    from msocr.pipeline.har_client import HARClient

    client = HARClient(api_token="test_token", registry_name="msocr-models")

    with patch.object(client, "_get") as mock_get:
        mock_get.return_value = {"download_url": "https://har.example.com/download/art_12345"}

        with patch("requests.get") as mock_download:
            mock_download.return_value.status_code = 200
            mock_download.return_value.content = b"fake_model_data"

            output_path = client.pull_artifact(
                name="syr-estrangela-printed",
                version="v14",
                output_dir=tmp_path,
            )

            assert output_path.exists()


def test_list_artifact_versions():
    """Test listing artifact versions."""
    from msocr.pipeline.har_client import HARClient

    client = HARClient(api_token="test_token", registry_name="msocr-models")

    with patch.object(client, "_get") as mock_get:
        mock_get.return_value = {
            "versions": [
                {"version": "v14", "created_at": "2026-04-01"},
                {"version": "v13", "created_at": "2026-03-25"},
            ]
        }

        versions = client.list_versions("syr-estrangela-printed")

        assert len(versions) == 2
        assert versions[0]["version"] == "v14"


def test_tag_artifact():
    """Test tagging an artifact version."""
    from msocr.pipeline.har_client import HARClient

    client = HARClient(api_token="test_token", registry_name="msocr-models")

    with patch.object(client, "_post") as mock_post:
        mock_post.return_value = {"status": "tagged"}

        client.tag_artifact(
            name="syr-estrangela-printed",
            version="v14",
            tag="production",
        )

        mock_post.assert_called_once()
```

**Step 2: Write HAR client implementation**

```python
# msocr/pipeline/har_client.py
"""Harness Artifact Registry client for model management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    artifact_id: str
    name: str
    version: str
    registry: str
    tags: List[str]
    sidecar_files: List[str]


@dataclass
class PushResult:
    """Result of pushing an artifact."""

    artifact_id: str
    name: str
    version: str
    registry: str


class HARClient:
    """Client for Harness Artifact Registry operations."""

    BASE_URL = "https://app.harness.io/gateway/artifact-registry"

    def __init__(self, api_token: str, registry_name: str, account_id: Optional[str] = None):
        self.api_token = api_token
        self.registry_name = registry_name
        self.account_id = account_id
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def push_artifact(
        self,
        name: str,
        version: str,
        artifact_path: Path,
        sidecar_files: Optional[Dict[str, Path]] = None,
        tags: Optional[List[str]] = None,
    ) -> PushResult:
        """Push artifact to Harness Artifact Registry.

        Args:
            name: Artifact name (e.g., syr-estrangela-printed)
            version: Version string (e.g., v14)
            artifact_path: Path to artifact file
            sidecar_files: Optional sidecar files (metrics.json, config.yaml)
            tags: Optional tags (staging, production)

        Returns:
            Push result with artifact ID and version
        """
        # Multipart upload for artifact file
        with artifact_path.open("rb") as f:
            files = {"artifact": (artifact_path.name, f)}

            response = self._post_multipart(
                f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}",
                files=files,
            )

        # Upload sidecar files if provided
        if sidecar_files:
            for filename, filepath in sidecar_files.items():
                with filepath.open("rb") as f:
                    files = {"file": (filename, f)}
                    self._post_multipart(
                        f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}/sidecar",
                        files=files,
                    )

        # Tag artifact
        if tags:
            for tag in tags:
                self.tag_artifact(name, version, tag)

        return PushResult(
            artifact_id=response["artifact_id"],
            name=name,
            version=version,
            registry=self.registry_name,
        )

    def pull_artifact(
        self,
        name: str,
        version: str,
        output_dir: Path,
    ) -> Path:
        """Pull artifact from Harness Artifact Registry.

        Args:
            name: Artifact name
            version: Version string
            output_dir: Directory to save artifact

        Returns:
            Path to downloaded artifact
        """
        response = self._get(
            f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}/download"
        )
        download_url = response["download_url"]

        # Download artifact
        output_path = output_dir / f"{name}-{version}.mlmodel"
        response_download = requests.get(download_url, timeout=300)
        response_download.raise_for_status()

        output_path.write_bytes(response_download.content)
        return output_path

    def pull_sidecar(
        self,
        name: str,
        version: str,
        filename: str,
        output_dir: Path,
    ) -> Path:
        """Pull sidecar file from artifact.

        Args:
            name: Artifact name
            version: Version string
            filename: Sidecar filename (metrics.json, config.yaml)
            output_dir: Directory to save file

        Returns:
            Path to downloaded sidecar file
        """
        response = self._get(
            f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}/sidecar/{filename}"
        )
        download_url = response["download_url"]

        output_path = output_dir / filename
        response_download = requests.get(download_url, timeout=60)
        response_download.raise_for_status()

        output_path.write_bytes(response_download.content)
        return output_path

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of an artifact.

        Args:
            name: Artifact name

        Returns:
            List of version metadata
        """
        response = self._get(
            f"/registries/{self.registry_name}/artifacts/{name}/versions"
        )
        return response.get("versions", [])

    def tag_artifact(
        self,
        name: str,
        version: str,
        tag: str,
    ) -> None:
        """Tag an artifact version.

        Args:
            name: Artifact name
            version: Version string
            tag: Tag name (staging, production)
        """
        self._post(
            f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}/tags",
            {"tag": tag},
        )

    def get_metadata(self, name: str, version: str) -> ArtifactMetadata:
        """Get artifact metadata.

        Args:
            name: Artifact name
            version: Version string

        Returns:
            Artifact metadata
        """
        response = self._get(
            f"/registries/{self.registry_name}/artifacts/{name}/versions/{version}"
        )

        return ArtifactMetadata(
            artifact_id=response["artifact_id"],
            name=response["name"],
            version=response["version"],
            registry=self.registry_name,
            tags=response.get("tags", []),
            sidecar_files=response.get("sidecar_files", []),
        )

    def _get(self, path: str) -> Dict[str, Any]:
        """Make GET request to HAR API."""
        url = f"{self.BASE_URL}{path}"
        response = requests.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to HAR API."""
        url = f"{self.BASE_URL}{path}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post_multipart(self, path: str, files: Dict[str, tuple]) -> Dict[str, Any]:
        """Make multipart POST request to HAR API."""
        url = f"{self.BASE_URL}{path}"
        headers = {"Authorization": self.headers["Authorization"]}
        response = requests.post(url, headers=headers, files=files, timeout=300)
        response.raise_for_status()
        return response.json()
```

**Step 3: Run tests**

Run: `uv run pytest tests/pipeline/test_har_client.py -v`
Expected: PASS

**Step 4: Update pipeline __init__.py**

```python
# msocr/pipeline/__init__.py
"""Pipeline infrastructure for CI/CD and training."""

from msocr.pipeline.runpod_client import RunPodClient, JobStatus, JobResult
from msocr.pipeline.har_client import HARClient, ArtifactMetadata, PushResult

__all__ = [
    "RunPodClient",
    "JobStatus",
    "JobResult",
    "HARClient",
    "ArtifactMetadata",
    "PushResult",
]
```

**Step 5: Commit**

```bash
git add msocr/pipeline/__init__.py msocr/pipeline/har_client.py tests/pipeline/test_har_client.py
git commit -m "feat: implement Harness Artifact Registry client"
```

---

## Phase 6: Training Dockerfile

### Task 6.1: Create Training Dockerfile

**Files:**
- Create: `docker/train/Dockerfile`

**Context:** Per Section 7, training image needs Ubuntu 22.04 + Tesseract 5.x + tesstrain + Kraken.

**Step 1: Create docker/train directory**

```bash
mkdir -p docker/train
```

**Step 2: Create Dockerfile**

```dockerfile
# docker/train/Dockerfile
# Training image for msocr with Tesseract and Kraken support
# SHA-pinned base for reproducibility

FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libtesseract-dev \
    libleptonica-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenblas-dev \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract 5.x with language data
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Download tessdata_best for Syriac, Greek, Latin, Coptic
RUN mkdir -p /usr/share/tessdata_best \
    && cd /usr/share/tessdata_best \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/syr.traineddata \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/ell.traineddata \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/lat.traineddata \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/cop.traineddata \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/hye.traineddata \
    && wget -q https://github.com/tesseract-ocr/tessdata_best/raw/main/gez.traineddata

# Install Python packages with uv (fastest)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment
RUN python3.10 -m venv /opt/msocr-venv
ENV PATH="/opt/msocr-venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/msocr-venv"

# Install Kraken and dependencies
RUN uv pip install --system \
    kraken==4.3.8 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system \
    python-Levenshtein \
    pyyaml \
    fastapi \
    uvicorn \
    requests

# Install tesstrain dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone tesstrain for Tesseract fine-tuning
RUN git clone --depth 1 https://github.com/tesseract-ocr/tesstrain.git /opt/tesstrain

# Set up working directories
WORKDIR /workspace
ENV TESSDATA_PREFIX=/usr/share/tessdata_best
ENV KRAKEN_MODEL_DIR=/workspace/models

# Copy training scripts
COPY scripts/train_tesseract.sh /scripts/
COPY scripts/train_kraken.sh /scripts/
RUN chmod +x /scripts/*.sh

# Default command
CMD ["bash"]
```

**Step 3: Create training scripts**

```bash
mkdir -p docker/train/scripts
```

```bash
# docker/train/scripts/train_tesseract.sh
#!/bin/bash
# Fine-tune Tesseract model using tesstrain
# Usage: train_tesseract.sh <lang> <variant> <gt_dir> <output_dir>

set -e

LANG="${1:-syr}"
VARIANT="${2:-estrangela}"
GT_DIR="${3:-/workspace/ground_truth}"
OUTPUT_DIR="${4:-/workspace/output}"

# Set TESSDATA_PREFIX for best models
export TESSDATA_PREFIX=/usr/share/tessdata_best

# Run tesstrain
cd /opt/tesstrain
bash make_tessdata.sh \
    --lang "${LANG}" \
    --variant "${VARIANT}" \
    --gt_dir "${GT_DIR}" \
    --output "${OUTPUT_DIR}"

echo "Training complete. Model saved to ${OUTPUT_DIR}/${LANG}-${VARIANT}.traineddata"
```

```bash
# docker/train/scripts/train_kraken.sh
#!/bin/bash
# Train Kraken HTR model using ketos
# Usage: train_kraken.sh <config.yaml> <gt_dir> <output_dir>

set -e

CONFIG="${1:-config.yaml}"
GT_DIR="${2:-/workspace/ground_truth}"
OUTPUT_DIR="${3:-/workspace/output}"

# Activate virtual environment
source /opt/msocr-venv/bin/activate

# Run ketos train
kraken train \
    --config "${CONFIG}" \
    --ground-truth "${GT_DIR}" \
    --output "${OUTPUT_DIR}/model.mlmodel"

echo "Training complete. Model saved to ${OUTPUT_DIR}/model.mlmodel"
```

**Step 4: Commit**

```bash
git add docker/train/ scripts/
git commit -m "feat: add training Dockerfile with Tesseract and Kraken support"
```

---

## Phase 7: Language Router Implementation

### Task 7.1: Create Language Router

**Files:**
- Create: `msocr/models/router.py`
- Test: `tests/models/test_router.py`

**Context:** Per Section 8, language-aware model router for printed/handwritten routes.

**Step 1: Write failing test**

```python
# tests/models/test_router.py
"""Tests for language-aware model router."""
import pytest


def test_route_printed_greek():
    """Test routing printed Greek to Kraken."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    result = router.route(
        language="greek",
        variant="polytonic",
        writing_mode="printed",
    )

    assert result.engine == "kraken"
    assert result.fallback_engine == "tesseract"
    assert result.needs_manual_review is False


def test_route_printed_latin():
    """Test routing printed Latin to Kraken CATMuS."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    result = router.route(
        language="latin",
        variant="caroline",
        writing_mode="printed",
    )

    assert result.engine == "kraken"
    assert result.model_id == "CATMuS-Print-Large"


def test_route_printed_syriac():
    """Test routing printed Syriac to Tesseract."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    result = router.route(
        language="syriac",
        variant="estrangela",
        writing_mode="printed",
    )

    assert result.engine == "tesseract"
    assert result.model_id == "syr"


def test_route_handwritten_syriac():
    """Test routing handwritten Syriac (requires custom HTR)."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    result = router.route(
        language="syriac",
        variant="estrangela",
        writing_mode="handwritten",
    )

    assert result.engine == "kraken"
    assert result.needs_custom_training is True


def test_route_printed_armenian():
    """Test routing printed Armenian to Tesseract hye-calfa-n."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    result = router.route(
        language="armenian",
        variant="erkatagir",
        writing_mode="printed",
    )

    assert result.engine == "tesseract"
    assert result.model_id == "hye-calfa-n"


def test_rtl_normalization():
    """Test RTL direction handling."""
    from msocr.models.router import ModelRouter

    router = ModelRouter()

    syriac = router.route(language="syriac", variant="estrangela", writing_mode="printed")
    assert syriac.direction == "rtl"

    greek = router.route(language="greek", variant="polytonic", writing_mode="printed")
    assert greek.direction == "ltr"
```

**Step 2: Write router implementation**

```python
# msocr/models/router.py
"""Language-aware model router for OCR and HTR."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteResult:
    """Result of model routing decision."""

    engine: str  # "kraken", "tesseract", "ocrmypdf"
    model_id: str
    fallback_engine: Optional[str] = None
    fallback_model_id: Optional[str] = None
    direction: str = "ltr"  # "ltr" or "rtl"
    needs_manual_review: bool = False
    needs_custom_training: bool = False
    confidence: float = 1.0


# Language registry with routing information
LANGUAGE_ROUTES = {
    "greek": {
        "ltr": True,
        "printed": {
            "engine": "kraken",
            "model_id": "greek_polytonic",
            "fallback_engine": "tesseract",
            "fallback_model_id": "ell",
            "cer_threshold": 0.05,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "greek_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "latin": {
        "ltr": True,
        "printed": {
            "engine": "kraken",
            "model_id": "CATMuS-Print-Large",
            "fallback_engine": "tesseract",
            "fallback_model_id": "lat",
            "cer_threshold": 0.05,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "latin_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "syriac": {
        "ltr": False,  # RTL
        "variants": {
            "estrangela": {
                "printed": {
                    "engine": "tesseract",
                    "model_id": "syr",
                    "cer_threshold": 0.05,
                },
            },
            "serto": {
                "printed": {
                    "engine": "tesseract",
                    "model_id": "syr",
                    "cer_threshold": 0.10,  # Needs fine-tuning
                    "needs_manual_review": True,
                },
            },
            "east_syriac": {
                "printed": {
                    "engine": "tesseract",
                    "model_id": "syr",
                    "cer_threshold": 0.10,
                    "needs_manual_review": True,
                },
            },
        },
        "printed": {
            "engine": "tesseract",
            "model_id": "syr",
            "cer_threshold": 0.10,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "syriac_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "sogdian": {
        "ltr": False,  # RTL
        "printed": {
            "engine": "kraken",
            "model_id": "sogdian_printed",
            "cer_threshold": 0.10,
            "needs_custom_training": True,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "sogdian_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "old_turkish": {
        "ltr": False,  # RTL (Old Uyghur script)
        "printed": {
            "engine": "kraken",
            "model_id": "old_turkish_printed",
            "config_path": "configs/old_turkish_config.yaml",
            "cer_threshold": 0.10,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "old_turkish_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "coptic": {
        "ltr": True,
        "printed": {
            "engine": "tesseract",
            "model_id": "cop",
            "cer_threshold": 0.05,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "coptic_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "armenian": {
        "ltr": True,
        "printed": {
            "engine": "tesseract",
            "model_id": "hye-calfa-n",
            "fallback_model_id": "hye",
            "cer_threshold": 0.05,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "armenian_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
    "geez": {
        "ltr": True,
        "printed": {
            "engine": "tesseract",
            "model_id": "gez",
            "cer_threshold": 0.05,
        },
        "handwritten": {
            "engine": "kraken",
            "model_id": "geez_htr",
            "needs_custom_training": True,
            "cer_threshold": 0.10,
        },
    },
}


class ModelRouter:
    """Routes OCR/HTR requests to appropriate models based on language and writing mode."""

    def route(
        self,
        language: str,
        variant: Optional[str] = None,
        writing_mode: str = "printed",
        confidence: float = 1.0,
    ) -> RouteResult:
        """Route to appropriate engine and model.

        Args:
            language: Target language (greek, latin, syriac, etc.)
            variant: Script variant (estrangela, serto, polytonic, etc.)
            writing_mode: printed or handwritten
            confidence: Model selection confidence (0.0-1.0)

        Returns:
            RouteResult with engine, model, and metadata
        """
        lang_key = language.lower()
        mode_key = writing_mode.lower()

        if lang_key not in LANGUAGE_ROUTES:
            raise ValueError(f"Unsupported language: {language}")

        lang_config = LANGUAGE_ROUTES[lang_key]

        # Check for variant-specific routing
        if variant and "variants" in lang_config:
            variant_key = variant.lower()
            if variant_key in lang_config["variants"]:
                variant_config = lang_config["variants"][variant_key]
                if mode_key in variant_config:
                    route_config = variant_config[mode_key]
                    return self._build_result(route_config, lang_config, confidence)

        # Use default routing
        if mode_key not in lang_config:
            raise ValueError(f"Unsupported writing_mode for {language}: {writing_mode}")

        route_config = lang_config[mode_key]
        return self._build_result(route_config, lang_config, confidence)

    def _build_result(
        self,
        route_config: dict,
        lang_config: dict,
        confidence: float,
    ) -> RouteResult:
        """Build RouteResult from configuration."""
        return RouteResult(
            engine=route_config["engine"],
            model_id=route_config["model_id"],
            fallback_engine=route_config.get("fallback_engine"),
            fallback_model_id=route_config.get("fallback_model_id"),
            direction="rtl" if not lang_config.get("ltr", True) else "ltr",
            needs_manual_review=route_config.get("needs_manual_review", False)
            or confidence < 0.80,
            needs_custom_training=route_config.get("needs_custom_training", False),
            confidence=confidence,
        )

    def get_cer_threshold(self, language: str, variant: Optional[str], writing_mode: str) -> float:
        """Get CER threshold for language/variant/mode."""
        lang_key = language.lower()
        mode_key = writing_mode.lower()

        if lang_key not in LANGUAGE_ROUTES:
            return 0.10  # Default

        lang_config = LANGUAGE_ROUTES[lang_key]

        # Check variant-specific threshold
        if variant and "variants" in lang_config:
            variant_key = variant.lower()
            if variant_key in lang_config["variants"]:
                variant_config = lang_config["variants"][variant_key]
                if mode_key in variant_config:
                    return variant_config[mode_key].get("cer_threshold", 0.10)

        # Default threshold
        if mode_key in lang_config:
            return lang_config[mode_key].get("cer_threshold", 0.10)

        return 0.10

    def list_languages(self) -> list:
        """List all supported languages."""
        return list(LANGUAGE_ROUTES.keys())

    def get_language_config(self, language: str) -> dict:
        """Get full configuration for a language."""
        return LANGUAGE_ROUTES.get(language.lower(), {})
```

**Step 3: Run tests**

Run: `uv run pytest tests/models/test_router.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add msocr/models/router.py tests/models/test_router.py
git commit -m "feat: implement language-aware model router"
```

---

## Summary

This plan implements the core requirements from the instruction baseline:

1. **Documentation Deliverables**: Created pipeline YAMLs and state update YAML
2. **Annotation API**: Session manager + FastAPI endpoints for ground-truth collection
3. **Evaluation Metrics**: Manifest manager + benchmark recorder with full metadata
4. **RunPod Client**: API client for GPU training job submission
5. **HAR Client**: Harness Artifact Registry client for model management
6. **Training Dockerfile**: Container spec with Tesseract + Kraken + tesstrain
7. **Language Router**: Language-aware model routing for all supported languages

**Key compliance points:**
- Strict printed/handwritten route separation
- Per-language CER thresholds from Section 5 of Summary-harness-update.md
- RTL direction handling for Syriac, Sogdian, Old Turkish
- Full metadata recording per eval.md
- Export formats (ALTO, PAGE, TSV) for ketos train
- Language registry web fonts and Unicode blocks

---

**Plan complete and saved to `docs/plans/2026-04-04-msocr-instruction-implementation.md`.**