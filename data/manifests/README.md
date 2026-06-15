# Sogdian Manifest Registry

Store reproducible Sogdian manuscript HTR training manifests in this directory.

- Manifests are keyed by `manifest_id` and normally live at `data/manifests/<manifest_id>.json`.
- Keep split manifests in git; keep large images/XML data outside git or under external dataset storage.
- Split manifests must isolate `manuscript_id` across `train`, `validation`, and `holdout` partitions.
- This project only accepts `language: "sogdian"` and `writing_mode: "handwritten"`.

Minimal schema:

```json
{
  "manifest_id": "sogdian-manuscript-v1",
  "writing_mode": "handwritten",
  "language": "sogdian",
  "dvc_tracked": true,
  "base_dir": "dataset/ground_truth/sogdian",
  "partitions": {
    "train": [
      {
        "id": "ms001_line_0001",
        "xml_path": "ms001/line_0001.xml",
        "manuscript_id": "ms001"
      }
    ],
    "validation": [],
    "holdout": []
  }
}
```
