# Christian Sogdian Fragment HTR and CER-Reduction Plan — v1

**Date:** 2026-08-02  
**Version:** v1  
**Status:** Planning baseline; no implementation in this document  
**Scope:** Christian Sogdian written in Syriac script, specifically the C2/C2AV (modern E27) material

Companion evidence report: [Kraken Fragmented-Manuscript Limitations and C2 Image-Source Report](./2026-08-02-kraken-fragmented-manuscript-limitations-v1.md)

## 1. Executive decision

Do **not** treat a publication plate, physical fragment, manuscript row, and
Kraken recognition line as the same object.

The safe training unit for Kraken recognition is a **contiguous readable line
fragment**: an image polygon containing visible ink paired with a transcript of
exactly that visible ink. When a hole or destructive lacuna interrupts a row,
the two readable sides become separate Kraken samples. They remain linked by
project metadata so the final output can reconstruct the logical row and render
an explicit gap.

```text
publication plate / source photograph
                │
                ▼
      isolate physical fragments
                │
                ▼
   normalize each fragment independently
                │
                ▼
 whole-row band for annotation and row grouping
                │
                ▼
 split at destructive holes or unreadable stretches
        │                         │
        ▼                         ▼
 right readable segment    left readable segment
  Kraken line sample        Kraken line sample
        └────────────┬────────────┘
                     ▼
       same row_id + RTL fragment order
                     ▼
      structured output with explicit gap metadata
```

The whole-row band may preserve the blank gap for human context, but a baseline
used for training must not cross the gap.

## 2. Locked decisions

These decisions were confirmed for v1:

1. The available local sources are publication plate images; no local
   individual-fragment photographs are available.
2. Final output should be structured: readable fragments plus explicit gap
   metadata. A presentation layer may render `[gap]`, but `[gap]` is not a
   recognition target.
3. Logical-row relationships are human-confirmed. Automatic Y-proximity may
   suggest a `row_id`, but it must not silently decide one.
4. A partial line at a fragment edge is trainable when every visible glyph is
   complete and transcribed. Exclude the sample when a glyph is visibly cut or
   cannot be identified.
5. Use a Leiden underdot for a probable but uncertain reading. Do not train the
   recognizer to emit U+FFFD for an unidentified glyph; exclude that affected
   sample instead.
6. Manual correction of fragment boundaries, row grouping, baselines, and
   transcripts is acceptable for the first 30–50 plates.
7. `expected_lines` is not a required annotation. It is only an optional
   extraction and QA hint, not a Kraken training label.
8. Primary recognition metric: CER on visible transcribed characters. Gap
   detection/localization is measured separately.
9. Benchmark both fine-tuned BLLA and Orli. Neither is assumed to solve logical
   row grouping.
10. This plan does not cover national-script Sogdian.

## 3. Why recognition and layout must be separated

Kraken recognition training transforms a line image into text. It does not need
automatic page segmentation if manually verified line polygons or line crops
already exist. Therefore, useful recognition training can begin before a robust
full-page segmenter exists.

The project should maintain two related datasets:

| Dataset view | Unit | Purpose |
|---|---|---|
| Recognition view | Contiguous readable line fragment | Train and evaluate the HTR recognizer |
| Layout view | Full source image with physical-fragment regions, split baselines, and reading order | Train and evaluate BLLA/Orli and end-to-end inference |

This prevents weak automatic segmentation from corrupting recognition ground
truth while still preserving full-page layout data for later training.

### 3.1 Current measured baseline

The present dataset and reports impose several constraints on all later claims:

| Fact verified in the repository | Consequence for v1 |
|---|---|
| 12 C2AV plates are annotated; 87 of the 99 plate images are not | Annotation quality and scale are higher-leverage than another immediate hyperparameter run |
| Current XML contains 154 train lines / 1,682 raw code points, 5 validation lines / 47, and 19 holdout lines / 429 | The manifest description saying 148/15/19 is stale and must not be used for experiment provenance; normalized evaluation counts may differ (the historical c2av12 audit used 427) |
| The best reported c2av12 CER is 71.43%; train CER on c2av01 was 6.08% | The roughly 6% versus 60–71% train/unseen gap indicates severe manuscript-page overfitting |
| c2av12 has been used repeatedly to select and diagnose approaches | It is now a development benchmark, not a pristine final test |
| The 83.4% deletion diagnosis was measured on the older 77.75%-CER model | Recompute the error mix for each new model; do not attribute that exact percentage to the 71.43% model |
| Train lines average about 10.9 characters; c2av12 averages about 22.6 | Line-length shift may contribute to under-emission and must be reported by length stratum |
| Recognition evaluation uses human baselines/generated polygons | It measures recognizer quality, not reliable automatic full-page performance |
| Runtime uses preprocessing plus default BLLA | Production end-to-end CER remains unestablished until full-page evaluation exists |

The current `walk_style_group()` implementation also checks a manifest
`base_model_override` before the explicit `base_model_path`, despite its
docstring and comment stating that the explicit argument wins. Phase 1 must
either correct that precedence or record and verify the actual resolved model
path before any new GPU run. Otherwise an experiment can train from a different
base than its command line claims.

### 3.2 Seven-phase delivery plan

This is the complete CER-reduction program. Phase 2 is intentionally expanded
for fragmented manuscripts; recognition training is not expected to repair bad
image/geometry representation by itself.

| Phase | Goal | Required output and gate |
|---|---|---|
| **1. Measurement and provenance** | Establish an honest baseline | Correct manifest counts; hash model/data; complete 12 plate-level LOPO under one frozen recipe; preserve c2av12 as development data; reserve a new pristine test set |
| **2. Fragment-aware image and line preparation** | Convert damaged plates into valid training units | C2/E27 source concordance; physical-fragment crops; per-fragment deskew; split baselines; `row_id` and gap metadata; five-plate contact-sheet pilot |
| **3. Ground-truth scale and QA** | Increase consistent Christian Sogdian data | 30-plate layout milestone, then approximately 50 plates / 700–800 atomic line fragments; transcript and geometry reviewer sign-off |
| **4. Recognition fine-tuning** | Lower CER with known-good polygons | Controlled base-model, codec-resize, freeze/unfreeze, length-balance, and augmentation experiments; visible-character CER by plate and line type |
| **5. Layout fine-tuning** | Make full-page extraction less manual | Default versus fine-tuned BLLA and Orli benchmark on the same held-out pages; baseline, split/merge, clipping, and order metrics |
| **6. Structured inference** | Reassemble fragments without inventing lost text | RTL `row_id` grouping, text/gap parts, mask use, confidence, and `[gap]` presentation renderer |
| **7. Release validation** | Promote only a real generalization gain | Recognition-only, layout-only, and end-to-end results on untouched plates; model card, hashes, limitations, and rollback artifact |

#### Phase 1 experiment ledger

Every run must record:

- Kraken version and container digest;
- base model path, checksum, and resolved provenance;
- output model checksum;
- manifest and PAGE XML checksums;
- exact train/validation/holdout manuscript IDs;
- exact command/options, seed, and augmentation recipe;
- line and visible-character counts after filtering;
- evaluation crop source: human polygon, generated polygon, or automatic page
  segmentation;
- CER, insertions, deletions, substitutions, and output/reference length ratio.

The 12 existing plates can support development cross-validation, but no result
on them should be called the final test after the annotation and preprocessing
policy has been selected from their errors.

## 4. Source-image acquisition and provenance

### 4.1 Source priority

Use the best available source in this order:

1. **Institutional archival image** supplied by BBAW/Staatsbibliothek.
2. **Digitales Turfan-Archiv individual-fragment image** under E27 / `n`
   shelfmarks.
3. **Publication plate image** from Sims-Williams 1985, with physical fragments
   isolated before annotation.
4. **Historical Göttingen/Hamburg photograph or publication facsimile** for a
   lost fragment.

An institutional image should not automatically replace a plate crop. The
public DTA derivatives sampled during research were only about 600–1,220 pixels
on the long side. Compare actual glyph height, sharpness, contrast, and visible
surface before choosing the training source.

### 4.2 Build a C2/E27 concordance first

Create one provenance record per C2 unit with:

- C2 folio and side;
- modern `E27/...` identifier;
- current `n ...` shelfmark;
- former `MIK III ...` identifier where applicable;
- excavation/packet identifier such as `T II B 44[c]`;
- local publication plate and coordinates;
- DTA recto/verso URL when available;
- image dimensions and checksum;
- source type: archival, DTA derivative, publication plate, or historical
  photograph;
- rights/access note;
- lost/surviving status.

The 2012 VOHD catalogue is the authoritative mapping source. Examples verified
during research include:

| C2 unit | Modern/current identifier | Public source |
|---|---|---|
| Folio 22 | `E27/22 = n 482 = MIK III 68 = T II B 44[c]` | [DTA recto](https://turfan.bbaw.de/dta/n/images/n482rectototal.jpg), [verso](https://turfan.bbaw.de/dta/n/images/n482versototal.jpg) |
| Folio 61 | `E27/61 = n 493 = MIK III 79` | [DTA recto](https://turfan.bbaw.de/dta/n/images/n493rectototal.jpg), [verso](https://turfan.bbaw.de/dta/n/images/n493versototal.jpg) |
| Folio 51c | `E27/51c = n 494 = MIK III 80` | [DTA recto](https://turfan.bbaw.de/dta/n/images/n494rectototal.jpg) |
| Folio 53 | `E27/53 = n 495 = MIK III 81` | [DTA recto](https://turfan.bbaw.de/dta/n/images/n495rectototal.jpg) |

### 4.3 Request archival-resolution images

Before bulk annotation, contact BBAW Turfanforschung / Staatsbibliothek
Orientabteilung with the concordance and ask whether archival TIFFs or the
higher-resolution DTA derivatives are available for non-commercial HTR
research. Request a small sample first and compare it against the plate source.

Do not redistribute institutional images or derived datasets until the DTA use
conditions and current permission terms have been confirmed.

## 5. Ground-truth data model

### 5.1 Required entities

#### `source_image`

The institutional photograph or publication plate from which coordinates are
measured.

Required metadata:

- source identifier and URL/path;
- checksum and dimensions;
- folio/side;
- rights/provenance;
- image-quality assessment.

#### `physical_fragment`

A connected manuscript-support piece visible in the source image.

Required metadata:

- `fragment_id`;
- polygon in source coordinates;
- orientation and deskew transform;
- source shelfmark when known;
- whether placement on the publication plate is artificial;
- local normalized image path.

#### `logical_row`

A manuscript row as understood by the annotator, even if physically split.

Required metadata:

- `row_id`;
- page/fragment membership;
- top-to-bottom order;
- principal RTL direction;
- ordered list of readable line-fragment IDs;
- ordered list of gap objects.

#### `line_fragment`

The atomic Kraken recognition sample: one contiguous readable portion of a
logical row.

Required metadata:

- unique line ID;
- parent `row_id`;
- `fragment_index` in RTL reading order;
- physical `fragment_id`;
- baseline and boundary polygon;
- diplomatic visible-text transcript;
- `trainable` flag and exclusion reason;
- source-to-normalized coordinate transform.

#### `gap`

A structured editorial/layout object between readable line fragments.

Suggested fields:

- `gap_id`;
- parent `row_id`;
- left/right neighboring line-fragment IDs in reading order;
- type: hole, torn edge, abrasion, stain, unknown, or separation between
  physical pieces;
- source polygon when visible;
- known or unknown extent;
- display rendering, default `[gap]`;
- confidence and annotator note.

The gap object is not included in the Kraken `<Unicode>` transcript.

### 5.2 PAGE XML representation

Use one `<TextLine>` per contiguous readable line fragment. Store the project
relationship in PAGE `custom` metadata and a sidecar manifest so that tools
which ignore custom metadata can still train normally.

Conceptual mapping:

```xml
<TextLine id="row05-frag00"
  custom="readingDirection {value:right-to-left;} row {id:row05;} fragment {index:0;}">
  <Coords points="..."/>
  <Baseline points="..."/>
  <TextEquiv><Unicode>visible right-hand text</Unicode></TextEquiv>
</TextLine>

<TextLine id="row05-frag01"
  custom="readingDirection {value:right-to-left;} row {id:row05;} fragment {index:1;}">
  <Coords points="..."/>
  <Baseline points="..."/>
  <TextEquiv><Unicode>visible left-hand text</Unicode></TextEquiv>
</TextLine>
```

The exact PAGE custom grammar must be tested for round-trip preservation by
Kraken and the annotation UI before finalizing it. The sidecar manifest remains
the project source of truth for row/gap relationships.

## 6. Annotation policy for damaged rows

### 6.1 Decision table

| Image condition | Ground-truth action | Kraken training action |
|---|---|---|
| Small hole outside the ink band | Keep one line; optionally annotate damage | Include normally |
| Hole intersects the row but visible ink remains contiguous around it | Annotator decides whether one tight polygon can exclude the hole without clipping ink | Include only after crop QA |
| Hole separates the row into disconnected readable pieces | Create separate `line_fragment`s with the same `row_id` | Train each piece independently |
| Large blank interval inside one row | Split; create `gap` metadata | Never train one polygon across the interval |
| Faint but readable glyph | Transcribe diplomatically; add Leiden underdot when reading is probable | Include |
| Unidentifiable visible glyph | Mark sample non-trainable or split around it if geometrically safe | Exclude affected sample; do not teach U+FFFD |
| Fragment edge after a complete visible glyph | Transcribe visible partial line | Include |
| Glyph physically bisected by fragment edge | Record editorial metadata | Exclude affected sample |
| Entire area unreadable/destroyed | Draw `DamageZone`; no baseline or transcript | Exclude |
| Two publication-plate pieces believed to be one original row | Separate line samples, shared human-confirmed `row_id` | Train separately; reassemble later |

### 6.2 Whole-row bands are annotation aids only

The current exact-count row-band extractor may preserve blank gaps and show the
annotator the whole logical row. This is useful for:

- seeing alignment across a hole;
- proposing `row_id` relationships;
- displaying the expected reading sequence;
- preventing a detached component from being mistaken for another row.

It is **not** the final recognition crop when a destructive gap is present.

### 6.3 Do not annotate expected row counts

Kraken does not train from an expected number of rows:

- recognition training uses line image/transcript pairs;
- segmentation training uses baseline and region geometry;
- Orli uses baseline polylines in source-file order.

If an edition supplies row counts, store them as optional acquisition/QA
metadata. Otherwise derive the annotated count from distinct `row_id` values.

## 7. Preprocessing and annotation workflow

### Stage A — Source selection

1. Resolve the C2/E27 identity and side.
2. Compare DTA individual image against publication plate crop.
3. Select the source with the best readable glyph detail.
4. Record provenance, dimensions, checksum, and rights.
5. Never mix two source images of the same folio across train/test without
   grouping them as the same manuscript unit.

### Stage B — Physical-fragment isolation

1. Detect candidate physical fragments using connected components and spatial
   clustering.
2. Reject printed labels, rulers, plate captions, shelfmarks, and photographic
   borders.
3. Require human confirmation of each physical-fragment polygon.
4. Preserve source coordinates.
5. Produce one lossless normalized image per physical fragment.

### Stage C — Per-fragment normalization

1. Deskew each physical fragment independently; never deskew the publication
   plate as one object.
2. Preserve grayscale/RGB as the recognition source.
3. Generate binarized masks only for geometry suggestions.
4. Do not replace faint original ink with hard binarization for recognition
   until a measured ablation proves it beneficial.
5. Store forward and inverse coordinate transforms.

### Stage D — Row and line-fragment annotation

1. Generate BLLA, Orli, or row-band suggestions.
2. Human corrects fragment regions and baselines.
3. Human groups related pieces into a `row_id`.
4. Human orders pieces from the visual right toward the left for RTL reading.
5. Split every baseline at destructive gaps.
6. Draw a tight boundary around each readable line fragment.
7. Add `DamageZone` polygons for destroyed areas.
8. Enter only visible diplomatic text in each line fragment.
9. Mark every sample trainable/non-trainable with a reason.

### Stage E — Dataset compilation

1. Render contact sheets showing source, baseline, boundary, and extracted line
   image.
2. Reject clipped, mostly blank, neighboring-line, or label-contaminated crops.
3. Compile verified line fragments into a Kraken Arrow recognition dataset.
4. Keep PAGE XML files for BLLA/Orli layout training.
5. Preserve plate/folio grouping in all splits.

## 8. Annotation-tool requirements before scaling

The current UI has region/baseline editing and RTL reading-order support, but a
fragment-aware production workflow additionally needs:

1. `row_id` on each line;
2. group/ungroup selected line fragments into one logical row;
3. visible RTL fragment index within a row;
4. gap creation between grouped fragments;
5. `trainable` toggle and exclusion reason;
6. physical-fragment region type distinct from `DamageZone`;
7. source-image provenance and coordinate-transform metadata;
8. whole-row context view plus atomic line-fragment crop preview;
9. warnings when a baseline crosses a `DamageZone`;
10. warnings when a crop is mostly blank or intersects another row;
11. contact-sheet export for reviewer sign-off;
12. round-trip preservation of row/gap metadata through PAGE XML and session
    state.

Automatic grouping should remain advisory. A false grouping is a semantic
error that CER alone will not reveal.

## 9. Recognition training plan

### 9.1 Pilot dataset

Start with five representative plates:

- one plate with several detached physical fragments;
- one plate with long rows crossing holes;
- one dark/low-contrast plate;
- one relatively clean plate;
- one plate containing rare Sogdian Syriac characters.

Produce both the old whole-line representation and the new split-line-fragment
representation for this pilot. Compare extracted crops visually and measure
recognition CER with the same model recipe.

### 9.2 Training unit and transcript

- Input: tight contiguous line-fragment polygon/crop.
- Target: visible diplomatic Syriac-script transcription only.
- Direction: RTL, using the existing project normalization convention.
- Gap markers: excluded from the target.
- Empty or destroyed fragments: excluded.
- Uncertain but probable glyph: include with agreed diplomatic convention.
- Unknown/bisected glyph: exclude the affected sample.

Do not impose an arbitrary minimum line length in the pilot. Keep short but
unambiguous fragments, record their grapheme count, report them separately,
and cap any oversampling so numerous one- or two-character pieces do not
dominate training. Revisit a minimum-length rule only if the pilot shows that
these samples destabilize alignment or evaluation.

### 9.3 Augmentation

Use only augmentation that preserves image/text alignment:

- mild rotation, scale, translation, blur, contrast, and noise;
- geometric transformations applied to both image and geometry before crop, or
  applied after the final atomic line crop;
- no synthetic blank interval paired with unchanged text unless it is a
  separately designed robustness experiment;
- no lacuna token experiment in the primary training path.

Visually approve an augmentation contact sheet before every new recipe.

### 9.4 Recognition evaluation

Report:

- micro CER on visible characters;
- macro CER by plate;
- CER by intact line versus split line fragment;
- CER by line-fragment length;
- deletion/substitution/insertion counts;
- rare-character CER;
- output/reference length ratio;
- excluded-line count and reasons.

The first decision gate is whether split atomic crops reduce CER and deletion
rate relative to whole-row crops containing holes.

## 10. Layout analysis: BLLA versus Orli

### 10.1 Candidates

Benchmark four conditions on identical held-out plates:

1. default Kraken BLLA;
2. BLLA fine-tuned on split baselines;
3. Orli base model;
4. Orli fine-tuned on the same split-baseline pages.

Orli integrates with Kraken 7 and emits baselines in source reading order, but
it remains a line detector. It does not provide logical `row_id` grouping or
gap semantics.

### 10.2 Segmentation targets

- Each contiguous readable line fragment is one baseline target.
- No target baseline crosses a hole.
- `DamageZone` is a region/mask annotation, not a baseline.
- Physical-fragment regions constrain processing but are not assumed to be a
  class BLLA can learn reliably from 12 pages.
- PAGE element order must reflect intended RTL row/fragment reading order for
  Orli compilation because Orli uses source-file order.

### 10.3 Layout metrics

Do not evaluate BLLA/Orli only through final CER. Report:

- baseline precision, recall, and F1 using cBAD-style matching;
- split errors: one true line fragment predicted as several;
- merge errors: unrelated fragments predicted as one line;
- missed readable line fragments;
- false lines on labels, stains, edges, and bleed-through;
- boundary coverage/clipping rate;
- reading-order footrule/Kendall metric where available;
- logical-row grouping accuracy in the project postprocessor.

### 10.4 Decision gate

Select the layout model only if it improves baseline recall and split/merge
rates without increasing clipped line polygons. If neither model is reliable,
retain human-corrected or row-band-assisted extraction for recognition data and
treat full-page automatic HTR as a later milestone.

## 11. Gap handling at inference

The runtime output should preserve structure rather than flattening immediately
to one string.

Automatic cross-gap row association is a separate application task; neither
BLLA nor Orli supplies it. The first runtime version should behave
conservatively:

1. Within one normalized physical fragment, suggest a join only when two line
   pieces have compatible baseline angle, projected Y position, line height,
   and a damage polygon between their facing endpoints.
2. Resolve competing suggestions one-to-one and return the score/evidence.
3. Between independently mounted physical pieces, provide suggestions only;
   require human confirmation because plate position may be artificial.
4. Leave low-confidence pieces as separate rows rather than making a false
   semantic join.
5. Evaluate suggestions against human `row_id` and RTL `fragment_index` truth.

The initial training workflow does not depend on automatic association because
annotators provide those relationships. On unseen pages, known/human damage
masks can be used immediately; automatic gap localization requires a separately
validated region/layout model and must not be assumed from ordinary BLLA/Orli
line output.

Suggested hierarchy:

```json
{
  "row_id": "row05",
  "direction": "rtl",
  "parts": [
    {"kind": "text", "line_id": "row05-frag00", "text": "..."},
    {"kind": "gap", "gap_type": "hole", "extent": "unknown"},
    {"kind": "text", "line_id": "row05-frag01", "text": "..."}
  ]
}
```

Presentation renderers may produce `right text [gap] left text`. CER is computed
over the concatenated visible text fragments, while gap detection receives its
own metric.

Known `DamageZone` polygons should be converted to a segmentation mask at
inference. Verify mask polarity against the installed Kraken API before use.
The mask suppresses false line detection inside damage; it does not join lines
or create `row_id` relationships.

## 12. Dataset scale and split plan

### Milestone 1 — five-plate pilot

Purpose: validate the representation and annotation workflow before scaling.

Done when:

- row/gap round-trip works;
- split fragments extract cleanly;
- no baseline crosses a destructive hole;
- contact-sheet QA is complete;
- recognition comparison between whole-row and atomic fragments exists.

### Milestone 2 — 30 fragment-aware plates

Purpose: first credible BLLA/Orli fine-tuning and layout evaluation.

Requirements:

- human-corrected physical-fragment regions;
- human-confirmed split baselines and `row_id`s;
- plate-grouped train/validation/test split;
- damage and image-quality strata represented in each development fold;
- at least three untouched plates reserved from model selection.

### Milestone 3 — approximately 50 plates / 700–800 line fragments

Purpose: recognition improvement and robust end-to-end evaluation.

Reserve a pristine final holdout of at least 1,000 visible characters across
three or more plates. Do not use it for annotation-policy tuning, preprocessing
selection, model selection, or error diagnosis.

## 13. Evaluation framework

### 13.1 Recognition-only

Human-corrected atomic line-fragment polygons → recognizer.

Primary metric: visible-character CER. This answers whether the recognition
model can read correctly when geometry is known.

### 13.2 Layout-only

Full image → BLLA/Orli baselines and order, compared with human geometry.

Primary metrics: baseline F1, split/merge rate, boundary coverage, and reading
order.

### 13.3 Gap handling

Predicted damage/gap structure compared with human gap objects.

Metrics:

- gap detection precision/recall/F1;
- DamageZone IoU or coverage recall;
- correct association of neighboring line fragments;
- logical-row grouping accuracy.

### 13.4 End-to-end

Full image → fragment processing → layout → recognition → row reconstruction.

Report:

- visible-character CER;
- missed-row and missed-line-fragment rate;
- falsely recognized label/damage text;
- row-order accuracy;
- gap detection metrics;
- separate results for DTA individual images and publication plates.

## 14. CER-reduction experiment order

1. Correct stale manifest counts and base-model provenance.
2. Complete plate-level LOPO under one frozen baseline recipe.
3. Re-audit the current best model and inspect every evaluation crop.
4. Reserve new pristine plates before using their transcriptions for decisions.
5. Build the five-plate atomic-line pilot.
6. Compare whole-row-with-hole versus split-line-fragment recognition.
7. Accept one annotation/crop policy and freeze its version.
8. Annotate to 30 plates while preserving plate/folio split isolation.
9. Benchmark BLLA and Orli, base and fine-tuned.
10. Annotate toward 50 plates and retrain recognition.
11. Run one-variable recognition experiments: raw Sophro versus Jer36 if
    available; Kraken-recommended `--resize new` versus the historical recipe;
    line-length balancing; augmentation; then partial backbone unfreezing.
12. Do not revive weighted-lexicon correction until recognition CER is low
    enough that substitutions, rather than deletions, are the dominant useful
    target.
13. Promote a model only after both recognition-only and end-to-end metrics
    improve on untouched plates.

## 15. Stop conditions and gates

| Gate | Required evidence | If it fails |
|---|---|---|
| Source gate | Selected image has sufficient glyph resolution and provenance | Request archival image or use publication fallback with limitation recorded |
| Geometry gate | Contact sheets show complete ink, no neighboring rows, no destructive gap inside a training polygon | Correct/split/exclude sample |
| Pilot gate | Atomic-fragment CER or deletion rate improves over whole-row-gap representation | Revisit crop width, split policy, and transcript alignment before scaling |
| Layout gate | Fine-tuned BLLA or Orli improves baseline F1 and split/merge rates | Keep human/row-band extraction; do not claim full automation |
| Data gate | New plates improve fresh-holdout CER after 300+ added lines | Reassess base-model match, ground-truth consistency, and preprocessing |
| Promotion gate | CV and pristine holdout both improve, with no major rare-character regression | Do not replace current runtime model |

## 16. Explicit non-goals for v1

- Reconstructing missing language inside a physical gap.
- Training Kraken to output Leiden gap markers.
- Inferring exact missing character count from hole width.
- Treating U+FFFD as a recognition class.
- Assuming publication-plate position is original manuscript layout.
- Fully automatic logical-row grouping.
- Training a new damage detector before sufficient DamageZone annotation exists.
- Replacing Kraken recognition with a new architecture before the line-fragment
  representation is validated.

## 17. Immediate next actions after the implementation slice

The concordance, public-image acquisition, fragment-aware fields, editor support,
and safe training exporter are complete. The remaining work starts with human
ground truth rather than more automatic line generation:

1. Select five downloaded E27 images covering the strata in section 9.1; keep
   `n020rectototal` as the already inspected relatively clean example.
2. Review each generated physical-fragment overlay and mask, correcting the ROI
   manually where mount material is retained or parchment is clipped.
3. Draw one baseline and polygon per contiguous visible line fragment, transcribe
   visible text only, and human-confirm `rowId`, RTL `fragmentIndex`, and gaps.
4. Mark bisected/ambiguous edge glyphs non-trainable with an exclusion reason;
   do not create a baseline through a hole and do not enter `[gap]` in text.
5. Export both Full PAGE and Training PAGE, then inspect crop contact sheets and
   the training inclusion/exclusion audit before accepting the pilot.
6. Freeze image-set-level development and holdout assignments, then compare the
   current recognizer on old whole-row crops versus reviewed atomic fragments
   under the same recipe.
7. Only after reviewed geometry exists, run the BLLA-versus-Orli layout benchmark
   and use baseline F1, clipping, split/merge, and order metrics—not CER alone.
8. Request archival-resolution BBAW/Staatsbibliothek images for any selected
   source whose public JPEG fails the glyph-resolution gate.

## 18. Implementation status on 2026-08-02

The first implementation slice now enforces the plan's most important safety
boundaries:

- the confirmed 120 DTA images are downloaded locally with tracked provenance
  and checksum verification while the JPEGs remain gitignored;
- mounted DTA photos use physical-parchment masking rather than ink-component
  DBSCAN;
- deskewed fragments receive newly computed geometry masks;
- optional BLLA output is explicitly proposal-only and accompanied by padded
  crops, provenance, and a contact sheet;
- the annotation state stores `rowId`, explicit RTL `fragmentIndex`, training
  eligibility/exclusion reason, and structural gaps;
- the editor groups disconnected pieces without invoking its geometry-join
  operation and blocks joining a declared fragmented row across a hole;
- full PAGE XML round-trips the metadata while the separate Training PAGE
  compiler physically removes excluded, empty, baseline-less, and
  gap-crossing lines;
- `[gap]` is rejected in Kraken transcripts;
- remote training automatically compiles safe PAGE XML and writes an inclusion
  and exclusion audit;
- an explicit command-line base model now takes precedence over a stale
  manifest style-group override.

The first default-BLLA E27 pilot (`n020rectototal`) produced eight proposals
but no complete usable lines. This is a stop result for automatic Phase 2
line generation, not a reason to manufacture training labels. The next
data-producing action is manual annotation of the five-plate atomic-line pilot;
fine-tuned BLLA/Orli comparison starts only after that reviewed geometry exists.
