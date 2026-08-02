"""Reproducible acquisition for externally hosted manuscript images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E27_SOURCE_MANIFEST = (
    REPO_ROOT / "data" / "manifests" / "c2-e27-dta-sources-v1.json"
)


@dataclass(frozen=True)
class SourceImage:
    """One immutable remote image described by a source manifest."""

    image_set: str
    filename: str
    side: str
    bytes: int
    width: int
    height: int
    sha256: str


@dataclass(frozen=True)
class SourceManifest:
    """Validated source manifest and its resolved local destination."""

    manifest_id: str
    base_url: str
    base_dir: Path
    allowed_hosts: frozenset[str]
    images: tuple[SourceImage, ...]
    units: tuple[dict[str, Any], ...]


def _rows_as_dicts(
    data: dict[str, Any], fields_key: str, rows_key: str
) -> list[dict[str, Any]]:
    fields = data.get(fields_key)
    rows = data.get(rows_key)
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(item, str) for item in fields)
    ):
        raise ValueError(f"Source manifest must define non-empty {fields_key}")
    if not isinstance(rows, list):
        raise ValueError(f"Source manifest must define {rows_key}")
    decoded = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(fields):
            value_count = len(row) if isinstance(row, list) else "invalid"
            raise ValueError(
                f"Source manifest {rows_key}[{index}] has {value_count} "
                f"values for {len(fields)} fields"
            )
        decoded.append(dict(zip(fields, row)))
    return decoded


def load_source_manifest(
    path: Path | str = DEFAULT_E27_SOURCE_MANIFEST,
) -> SourceManifest:
    """Load and validate a compact external-image source manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_id = data.get("manifest_id")
    source = data.get("source") or {}
    base_url = source.get("base_url")
    base_dir = data.get("base_dir")
    allowed_hosts = frozenset(data.get("allowed_hosts") or [])
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("Source manifest missing manifest_id")
    if not isinstance(base_url, str) or urlparse(base_url).scheme != "https":
        raise ValueError("Source manifest source.base_url must be HTTPS")
    if not allowed_hosts or urlparse(base_url).hostname not in allowed_hosts:
        raise ValueError("Source manifest base URL host is not in allowed_hosts")
    if not isinstance(base_dir, str) or not base_dir:
        raise ValueError("Source manifest missing base_dir")

    image_rows = _rows_as_dicts(data, "image_fields", "images")
    unit_rows = _rows_as_dicts(data, "unit_fields", "units")
    images: list[SourceImage] = []
    filenames: set[str] = set()
    for row in image_rows:
        image = SourceImage(
            image_set=str(row["image_set"]),
            filename=str(row["filename"]),
            side=str(row["side"]),
            bytes=int(row["bytes"]),
            width=int(row["width"]),
            height=int(row["height"]),
            sha256=str(row["sha256"]),
        )
        if Path(image.filename).name != image.filename:
            raise ValueError(f"Unsafe source filename: {image.filename}")
        if image.filename in filenames:
            raise ValueError(f"Duplicate source filename: {image.filename}")
        if len(image.sha256) != 64:
            raise ValueError(f"Invalid SHA-256 for {image.filename}")
        if min(image.bytes, image.width, image.height) <= 0:
            raise ValueError(f"Invalid image metadata for {image.filename}")
        filenames.add(image.filename)
        images.append(image)

    image_sets = {image.image_set for image in images}
    unit_ids: set[str] = set()
    for unit in unit_rows:
        unit_id = str(unit.get("e27_id") or "")
        if not unit_id or unit_id in unit_ids:
            raise ValueError(f"Invalid or duplicate E27 unit: {unit_id!r}")
        status = unit.get("status")
        image_set = unit.get("image_set")
        if status not in {"surviving", "lost"}:
            raise ValueError(f"Invalid status for {unit_id}: {status!r}")
        if status == "surviving" and image_set not in image_sets:
            raise ValueError(
                f"Surviving unit {unit_id} references unknown image set {image_set!r}"
            )
        if status == "lost" and image_set is not None:
            raise ValueError(f"Lost unit {unit_id} must not reference an image set")
        unit_ids.add(unit_id)

    resolved_dir = Path(base_dir)
    if not resolved_dir.is_absolute():
        resolved_dir = REPO_ROOT / resolved_dir
    return SourceManifest(
        manifest_id=manifest_id,
        base_url=base_url,
        base_dir=resolved_dir,
        allowed_hosts=allowed_hosts,
        images=tuple(images),
        units=tuple(unit_rows),
    )


def _verify_image(path: Path, expected: SourceImage) -> None:
    data = path.read_bytes()
    if len(data) != expected.bytes:
        raise ValueError(
            f"Size mismatch for {path.name}: expected {expected.bytes}, got {len(data)}"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected.sha256}, got {digest}"
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = image.size
    except Exception as exc:
        raise ValueError(f"Unreadable image {path.name}: {exc}") from exc
    if size != (expected.width, expected.height):
        raise ValueError(
            f"Dimension mismatch for {path.name}: expected "
            f"{expected.width}x{expected.height}, got {size[0]}x{size[1]}"
        )


def acquire_source_images(
    manifest_path: Path | str = DEFAULT_E27_SOURCE_MANIFEST,
    *,
    output_dir: Path | None = None,
    verify_only: bool = False,
    force: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Download or verify every image in a source manifest.

    Existing valid files are never fetched again. Existing invalid files fail
    closed unless ``force`` is set, preventing silent replacement of source
    material when an upstream derivative changes.
    """
    manifest = load_source_manifest(manifest_path)
    destination = Path(output_dir) if output_dir is not None else manifest.base_dir
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    verified = 0

    for image in manifest.images:
        target = destination / image.filename
        if target.exists() and not force:
            _verify_image(target, image)
            verified += 1
            continue
        if verify_only:
            if target.exists():
                _verify_image(target, image)
                verified += 1
                continue
            raise FileNotFoundError(f"Source image missing: {target}")

        url = urljoin(manifest.base_url, image.filename)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in manifest.allowed_hosts:
            raise ValueError(f"Refusing source URL outside allowed hosts: {url}")
        request = Request(
            url,
            headers={
                "User-Agent": "msocr-research/1.0 (non-commercial manuscript HTR research)"
            },
        )
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(request, timeout=timeout) as response:
                partial.write_bytes(response.read())
            _verify_image(partial, image)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        downloaded += 1

    return {
        "manifest_id": manifest.manifest_id,
        "directory": str(destination),
        "total": len(manifest.images),
        "downloaded": downloaded,
        "verified": verified + downloaded,
        "bytes": sum(image.bytes for image in manifest.images),
        "surviving_units": sum(
            unit["status"] == "surviving" for unit in manifest.units
        ),
        "lost_units": sum(unit["status"] == "lost" for unit in manifest.units),
    }
