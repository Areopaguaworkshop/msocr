# Frozen Manifest Registry

Store reproducible training and benchmark manifests in this directory.

- Manifests are keyed by `manifest_id` and should normally live at `data/manifests/<manifest_id>.json`.
- Track the underlying corpus with DVC and keep the manifest itself in git so a model or benchmark report can always be traced back to an immutable split.
- Split manifests must isolate `manuscript_id` across `train`, `validation`, and `holdout` partitions.

Minimal schema:

```json
{
  "manifest_id": "syriac-printed-v1",
  "writing_mode": "printed",
  "language": "syriac",
  "dvc_tracked": true,
  "base_dir": "dataset/ground_truth/syriac",
  "partitions": {
    "train": [
      {
        "id": "ms001_line_0001",
        "xml_path": "ms001/line_0001.xml",
        "manuscript_id": "ms001"
      }
    ],
    "validation": [],
    "holdout": [
      {
        "id": "ms002_case_0001",
        "image": "ms002/page_0001.png",
        "reference_text": "ms002/page_0001.txt",
        "language": "syriac",
        "script_variant": "estrangela",
        "manuscript_id": "ms002"
      }
    ]
  }
}
```
