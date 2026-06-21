# Fragmented Manuscript Line Segmentation Strategy

Goal: produce the exact number of manuscript text-line crops for annotation, even when one line is fragmented into left/right pieces with a blank middle.

## Policy

- Count manuscript text rows, not connected components.
- Keep fragmented pieces on the same row in one crop.
- Preserve blank gaps inside a row crop.
- Exclude shelfmarks, page labels, isolated flecks, and publication labels.
- Make the expected line count explicit, e.g. `--expected-lines 18`.

## Pipeline

1. Start with Kraken BLLA/page geometry when it is useful, but do not trust component-level splits as final lines.
2. Threshold ink and find connected ink components.
3. Drop tiny flecks by component area.
4. Cluster component vertical centers into exactly the requested number of rows.
5. Crop one padded line band per row from the min/max component extent in that row.
6. Save review artifacts:
   - numbered line crops
   - contact sheet
   - overlay image with numbered boxes on the source page
7. Load the resulting crops into the annotation UI for manual transcription and correction.

If automatic clustering merges or splits rows, switch to row-center mode. This mirrors eScriptorium correction practice: a human marks the intended baselines/row centers, and the pipeline uses those anchors to make exactly one crop per manuscript row.

## Required Human Decision

Exact line counts require a page text region. For this C2AV image, the expected count is 18 manuscript rows. Labels and shelfmarks must be excluded by ROI or by manual rejection; they are text-like ink and cannot be reliably distinguished by a generic line segmenter.

## Command Shape

```bash
uv run msocr extract-lines \
  tmp/pdfs/ms_c2av_image19/page-19.png \
  --expected-lines 18 \
  --output-dir dataset/smoke/christian_sogdian_c2av_image19/extracted-lines \
  --roi LEFT,TOP,RIGHT,BOTTOM
```

`--roi` is optional, but recommended for scans with page labels or shelfmarks.

For corrected extraction:

```bash
uv run msocr extract-lines \
  tmp/pdfs/ms_c2av_image19/page-19.png \
  --expected-lines 18 \
  --output-dir dataset/smoke/christian_sogdian_c2av_image19/extracted-lines \
  --roi LEFT,TOP,RIGHT,BOTTOM \
  --row-centers Y1,Y2,Y3,...,Y18
```
