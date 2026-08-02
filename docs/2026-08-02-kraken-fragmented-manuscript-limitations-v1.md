# Kraken Fragmented-Manuscript Limitations and C2 Image-Source Report — v1

**Date:** 2026-08-02  
**Version:** v1  
**Status:** Evidence report supporting the Christian Sogdian fragment HTR plan  
**Scope:** Kraken 7.x layout/recognition behavior, Orli, and online source images for Christian Sogdian C2/C2AV (E27)

Companion plan: [Christian Sogdian Fragment HTR and CER-Reduction Plan](./2026-08-02-christian-sogdian-fragment-htr-plan-v1.md)

## 1. Summary

Kraken recognition can be trained effectively on manually verified line
fragments even when automatic full-page segmentation is not yet reliable.
Kraken does **not**, however, natively represent the philological relationship
between disconnected pieces of one manuscript row. That relationship and the
gap between pieces must be stored and reconstructed by msocr.

The safest representation is therefore:

- one Kraken line sample per contiguous readable ink segment;
- no baseline crossing a destructive hole;
- no gap token in recognition ground truth;
- project-level `row_id`, RTL fragment order, and gap metadata;
- separate evaluation of recognition, layout, gap localization, and end-to-end
  output.

Online research also found public individual-fragment images for many surviving
C2 units in the BBAW/Staatsbibliothek Digitales Turfan-Archiv under the modern
designation E27. The public images are useful but often modest web derivatives;
archival higher-resolution files likely require an institutional request.

## 2. Research method and evidence quality

This report checked:

- Kraken 6.0 documentation where current public documentation routes were
  available;
- current Kraken GitHub source and upstream issues as of 2026-08-02;
- the locally installed/project-targeted Kraken 7.0.2 behavior documented in
  this repository;
- the current Orli repository and model documentation;
- the BBAW Digitales Turfan-Archiv;
- the 2012 VOHD catalogue concordance for Iranian manuscripts in Syriac script;
- CrossAsia, IDP, Staatsbibliothek, and Museum für Asiatische Kunst portals;
- Zenodo, GitHub, and general repository searches for a separate C2 image
  dataset.

Important documentation caveat: several links in older project documents use
`kraken.re/main/...`, which currently returns 404. The Kraken 6.0 pages remain
available, while current 7.x implementation details were checked in source and
against local 7.0.2 notes. Version-sensitive commands must be verified against
the installed CLI before execution.

## 3. Confirmed Kraken limitations

### 3.1 The default BLLA model targets undegraded, even writing surfaces

Kraken's page-segmentation documentation describes the default baseline model
as working on printed and handwritten material on **undegraded, even writing
surfaces such as paper or parchment**. Torn Turfan fragments with holes,
darkened support, stains, abrasion, detached pieces, and artificial publication
layouts are outside that stated base distribution.

Implication: default BLLA output is an annotation suggestion, not trustworthy
ground truth for C2.

Source: [Kraken page segmentation documentation](https://kraken.re/6.0.0/advanced/segmentation.html).

### 3.2 BLLA does not understand logical manuscript rows

BLLA labels pixels as line/region classes, vectorizes the heatmap into line
instances and region boundaries, computes line polygons, and then assigns
reading order. It has no semantic object meaning “these two disconnected
baselines belong to one original row.”

Implication: logical-row grouping must be represented by msocr metadata and
human confirmation. It cannot be recovered reliably from the BLLA line list.

### 3.3 Holes and degradation commonly produce split lines

[Kraken issue #745](https://github.com/mittagessen/kraken/issues/745) reports
exactly this failure mode on historical manuscripts: darkened support, faded or
erased ink, holes, and interlinear text interrupt horizontal detection and
produce multiple detected segments for one manuscript line. The requested
“force vertical-only segmentation” or automatic joining was not added. The
maintainer recommended fine-tuning the base segmentation model and noted that a
new layout system was intended to replace BLLA.

[Kraken issue #677](https://github.com/mittagessen/kraken/issues/677) provides a
second report of broken/split lines persisting across several BLLA training
recipes. A differently matched segmentation base model helped, showing that
model/domain match matters, but there is no general joining guarantee.

Implication: never assume BLLA's disconnected outputs are distinct logical
rows, and never automatically join them without row-level evidence.

### 3.4 A baseline crossing a hole creates a false segmentation target

Current Kraken segmentation-dataset source converts each annotated baseline
polyline into a buffered continuous target shape and marks start/end separator
classes. There is no discontinuity operator inside one baseline.

If an annotation draws one baseline across a blank hole, the training target
explicitly says that baseline pixels should exist through the hole. This
contradicts the image and teaches an unstable target.

Implication: split a damaged logical row into separate baseline targets at the
hole.

Source: [`BaselineSet.transform()` in Kraken](https://github.com/mittagessen/kraken/blob/67df2800cf470fa7d84017938e2f1d1ef5679299/kraken/lib/dataset/segmentation.py).

### 3.5 There is no per-pixel ignore channel for recognition training

Kraken recognition datasets load one line image and one text target. The API
supports text normalization, bidi reordering, augmentation, and filtering of
empty lines, but no target mask that says “ignore the middle third of this
line.”

Implication: a polygon containing a large blank lacuna remains an ordinary
sequence paired with the complete transcript. The robust primary solution is
to split it into contiguous samples, not to expect the loss to ignore the gap.

Source: [Kraken recognition dataset source](https://github.com/mittagessen/kraken/blob/67df2800cf470fa7d84017938e2f1d1ef5679299/kraken/lib/dataset/recognition.py).

### 3.6 `skip_empty_lines` does not solve internal holes

`skip_empty_lines` filters a sample when its transformed **text target is
empty**. It does not inspect or remove blank spans inside an image whose target
contains text.

Implication: it protects against wholly empty GT records, not against a line
polygon crossing a hole.

### 3.7 A segmentation mask is inference-only suppression

Kraken accepts a page-sized binary mask that suppresses line detection in
selected areas. It can stop the segmenter from proposing lines inside known
damage or labels.

It does not:

- join two line fragments across a gap;
- create logical-row relationships;
- remove an internal gap from a recognition crop;
- repair training targets;
- infer missing text.

The public documentation contains wording that can be read inconsistently with
respect to mask polarity, so polarity must be verified with the installed
Kraken API before integration.

Source: [Kraken masking documentation](https://kraken.re/6.0.0/advanced/segmentation.html#masking).

### 3.8 `DamageZone` does not automatically protect recognition training

PAGE XML region types may be parsed for layout/segmentation training, but
recognition training still consumes line geometry and text. Merely drawing a
`DamageZone` does not guarantee that a line polygon will exclude it.

Implication: dataset preparation must split the baseline, provide a tight line
boundary, exclude the sample, or explicitly apply a mask in the relevant
inference path.

### 3.9 Generated line polygons are CER-sensitive

[Kraken issue #773](https://github.com/mittagessen/kraken/issues/773) confirmed
a Kraken 7 beta regression where a smaller polygonization offset clipped
historical handwriting and increased CER. Restoring wider context restored the
previous CER. The upstream bug was fixed, so this is not a claim that current
Kraken 7.0.2 necessarily contains that regression; it is evidence that line
boundary context directly changes recognition results.

Implication: every dataset and release needs contact-sheet QA for clipped
ascenders, descenders, diacritics, fragment-edge glyphs, and neighboring-line
contamination.

### 3.10 BLLA masks and regions cannot encode missing language

Kraken layout models predict geometry, not editorial reconstruction. Neither a
`DamageZone` nor a mask gives the recognizer a principled number of missing
characters or a missing word.

Implication: the output needs a structured gap object. Missing language remains
unknown unless supplied by an independent editorial layer, which must not be
scored as visual recognition.

### 3.11 CTC can absorb blank intervals but cannot describe them

A recognition line with a large internal blank interval contains many image
timesteps with no visible glyph. Standard CTC can align those frames to its
internal blank label, so a hole does not make recognition mathematically
impossible. But the CTC blank is an alignment mechanism, not an output symbol:
it cannot describe the presence, extent, or type of a manuscript gap.

The current project's older 77.75%-CER audit found severe under-emission, but
that does not prove holes caused those deletions. Implication: compare whole-row
and split-fragment representations empirically, while using split contiguous
fragments as the safer, semantically correct default.

### 3.12 Automatic line joining is not equivalent to row reconstruction

Two baselines can be close in Y yet belong to different rows, fragments, sides,
or publication-plate arrangements. Conversely, pieces of one row may be
vertically displaced by artificial mounting or fragment rotation.

Implication: Y-proximity may generate a suggestion, but a scholar must confirm
`row_id` and RTL order for training/evaluation truth.

### 3.13 Evidence-to-action table

| Confirmed boundary | Primary evidence | Practical consequence | v1 mitigation |
|---|---|---|---|
| Default BLLA assumes a substantially cleaner surface | Kraken segmentation documentation and base-model description | Default output is out-of-domain on C2 | Use only as a proposal; fine-tune and compare with Orli |
| One baseline has continuous geometry | Kraken segmentation-dataset source | A cross-hole baseline creates a false target | Split at destructive lacunae |
| Recognition has no internal ignore target | Kraken recognition-dataset source | Blank gap and transcript remain one CTC sequence | Export contiguous line fragments independently |
| Holes trigger line splitting in practice | Kraken issues [#745](https://github.com/mittagessen/kraken/issues/745) and [#677](https://github.com/mittagessen/kraken/issues/677) | Predicted pieces cannot be assumed to be separate rows | Human-confirm `row_id`; score split/merge errors |
| Inference masks suppress rather than reconstruct | Kraken masking documentation | Mask cannot join pieces or infer text | Use mask only to avoid false layout detections |
| Damage regions do not alter recognition automatically | PAGE/layout versus recognition data contracts | `DamageZone` alone does not make a crop safe | Convert to a mask, split geometry, or exclude |
| Polygon context changes CER | Kraken issue [#773](https://github.com/mittagessen/kraken/issues/773) | Tight/generated boundaries can clip useful ink | Require extraction contact-sheet QA |
| Orli predicts baselines and order, not philological rows | Orli README/training format | Better layout does not create gap semantics | Benchmark it, but retain msocr row/gap metadata |

## 4. Orli: capabilities and limitations

[Orli](https://github.com/mittagessen/orli) is a published layout-analysis
system that integrates with Kraken 7 through the model-plugin mechanism. It
directly predicts text-line baselines in reading order and is designed to
replace a separate line-detection plus heuristic-ordering pipeline.

Confirmed useful properties:

- released base model trained on 200,000 pages across ten writing systems;
- direct baseline regression;
- explicit reading-order output;
- Kraken 7 inference integration;
- fine-tuning and cBAD-style baseline metrics;
- published base weights.

Confirmed operational constraints:

- the base model requires `bf16-mixed` precision according to its README;
- recommended high-resolution input is 1920×1440;
- training uses Orli-specific Arrow files, not Kraken recognition Arrow files;
- compilation takes line source-file order as reading order and ignores other
  reading-order annotations;
- no documented logical-row/gap object exists;
- no published evidence found that it joins disconnected pieces across physical
  holes in Turfan-style fragments;
- no evidence found that it removes the need for human-confirmed fragment
  grouping.

Conclusion: Orli should be benchmarked beside BLLA. It may improve baseline
detection and order, but it is not a substitute for the proposed `row_id` and
gap representation.

### 4.1 Avoid these misconceptions

The limitations above do **not** mean:

- Kraken is unusable for fragmentary manuscripts. Its recognizer can train on
  properly prepared contiguous line fragments.
- Every small blemish requires a split. Split where support/ink continuity and
  transcription alignment are materially broken; retain one crop when all
  visible ink can be bounded safely.
- Fine-tuning BLLA is pointless. It may substantially improve detection on C2,
  but it cannot supply the missing logical-row abstraction.
- Orli is known to fail on C2. Its C2 performance is unknown and must be
  benchmarked rather than assumed.
- A segmentation mask has no value. It is useful for suppressing damage,
  labels, and non-manuscript areas at inference, just not for training-target
  repair or row joining.
- Editorial reconstruction is forbidden. It belongs in a separate scholarly
  layer and must not be represented as visually recognized text.

## 5. What Kraken training can and cannot use

| Information | Recognition training | BLLA segmentation training | Orli training |
|---|---:|---:|---:|
| Line image/polygon | Required | Derived from page geometry | Page-level source |
| Transcript | Required | Not the baseline target | Optional for line detection |
| Baseline | Used to extract polygonal line | Required target | Required target |
| Region polygon | Not a recognition target | Optional region target/boundary context | Not the logical-row solution |
| `DamageZone` | No automatic effect | Can be a region class if configured | No confirmed gap semantics |
| Internal per-pixel ignore | No | No documented target ignore channel | No project-validated support |
| Expected line count | No | No | No |
| Logical `row_id` | Ignored unless msocr uses it | Not native | Not native |
| Gap marker/extent | Ordinary text if inserted, therefore risky | Not represented by a baseline | Not native |
| Reading order | Text bidi within line | Heuristic/model-dependent | Source element order during compile |

Answer to the project's explicit question: the project field
`expected_lines`, or any equivalent expected row count, **does not need to be
annotated for training**. It may remain an optional extraction/QA hint.

## 6. C2/E27 online image findings

### 6.1 Main finding

Many surviving physical fragments of Christian Sogdian manuscript C2 have
individual public images in the BBAW/Staatsbibliothek **Digitales
Turfan-Archiv (DTA)**. The manuscript is now designated **E27**, and many units
use shelfmarks in the `n 1–n 496` collection; former museum pieces also carry
`MIK III 61–82` identifiers.

Collection links:

- [Digitales Turfan-Archiv](https://turfan.bbaw.de/dta/)
- [Christian Sogdian `n` collection index](https://turfan.bbaw.de/dta/n/dta_n_index.html)
- [BBAW project context](https://turfan.bbaw.de/projekt-en.html)

These are individual-fragment or reconstructed-folio images rather than crops
from Sims-Williams's plate PDF.

### 6.2 Public image quality

Observed public examples:

| Image | Observed dimensions | Approximate size |
|---|---:|---:|
| `n493rectototal.jpg` | 792×606 | 54 KB |
| `n490versototal.jpg` | 1220×1137 | 131 KB |
| `n482rectototal.jpg` | 600×711 | 51 KB |

The DTA project description discusses higher-resolution images in its archival
workflow, but the old public interface appears to expose smaller derivatives.
Therefore, DTA images may offer cleaner isolation and provenance without always
offering more pixels than the local 2480×3509 publication plates.

Source: [DTA image workflow description](https://turfan.bbaw.de/dta-i-en.html).

### 6.3 Verified C2 mappings and URLs

#### Plate XIX / folio 22

The visible label `Pethion B 44 c` maps to:

- C2 folio 22;
- `E27/22`;
- `n 482`;
- `MIK III 68`;
- excavation/packet mark `T II B 44[c]`;
- text: *Martyrdom of St. Pethion*.

Images:

- [n 482 recto](https://turfan.bbaw.de/dta/n/images/n482rectototal.jpg)
- [n 482 verso](https://turfan.bbaw.de/dta/n/images/n482versototal.jpg)

#### Folio 61

- `E27/61 = n 493 = MIK III 79`.
- [n 493 recto](https://turfan.bbaw.de/dta/n/images/n493rectototal.jpg)
- [n 493 verso](https://turfan.bbaw.de/dta/n/images/n493versototal.jpg)

The BBAW project page explicitly identifies n 493 recto as belonging to the
large Sogdian codex C2.

#### Other verified examples

| Unit | Identifiers | Image |
|---|---|---|
| Folio 54 | `E27/54 = n 490 = MIK III 76 = T II B 16[a]` | [n 490 verso](https://turfan.bbaw.de/dta/n/images/n490versototal.jpg) |
| Folio 51c | `E27/51c = n 494 = MIK III 80 = T II B 67[d]` | [n 494 recto](https://turfan.bbaw.de/dta/n/images/n494rectototal.jpg) |
| Folio 53 | `E27/53 = n 495 = MIK III 81 = T II B 33[c]` | [n 495 recto](https://turfan.bbaw.de/dta/n/images/n495rectototal.jpg) |

### 6.4 Plate I

The first C2 folio comprises four non-joining pieces:

| Unit | Current shelfmark | Status |
|---|---|---|
| `E27/1a` | `n 69` | Surviving; DTA image expected |
| `E27/1b` | none | Lost |
| `E27/1c` | `n 70 + n 71` | Surviving; DTA images |
| `E27/1d` | `n 72` | Surviving; DTA image |

The lost piece means Plate I cannot be replaced completely by modern physical-
fragment images. The publication or historical photograph remains necessary
for `E27/1b`.

### 6.5 Plate XI / folio 12

The plate label mentioning a Hamburg photograph reflects historical source
photography. Folio 12 includes:

| Unit | Identifier | Status/note |
|---|---|---|
| `E27/12a` | `n 479 = MIK III 65 = T II B 66[e]` | Surviving; historical Göttingen/Hamburg photographs also recorded |
| `E27/12b` | `n 74` | Surviving DTA fragment |
| `E27/12c` | `n 73 = T II B 60` | Surviving DTA fragment |

The historical photograph may preserve a different pre-restoration state from
the modern DTA image. Both should be retained as provenance variants, never
split across training and test as independent examples.

### 6.6 Lost/photo-only units

The 2012 catalogue lists at least these E27 units as lost:

`E27/1b`, `E27/31`, `E27/40c`, `E27/51a`, `E27/55b`, `E27/57b`,
`E27/69b`, `E27/94a`, `E27/94b`, and `E27/108`.

Some survive in older Göttingen/Hamburg photographs or publication plates.
These must be tagged as historical-photo/publication-only sources rather than
modern fragment photographs.

### 6.7 Authoritative catalogue

The main concordance source is Nicholas Sims-Williams, *Iranian Manuscripts in
Syriac Script in the Berlin Turfan Collection*, VOHD XVIII/4 (2012):

- [Catalogue PDF](https://rep.adw-goe.de/bitstream/handle/11858/00-001S-0000-0023-9AE7-E/687736323.pdf?sequence=1&isAllowed=y)

It maps C2 folio numbers to E27 units, `n` shelfmarks, `MIK III` identifiers,
excavation packet marks, and lost/photo-only fragments.

## 7. Other searched image sources

### 7.1 CrossAsia German Turfan Expeditions

[CrossAsia's expedition portal](https://themen.crossasia.org/deutsche-turfanexpeditionen/?lang=en)
contains institutional expedition records, drawings, and more than 3,000 glass
negatives, some with IIIF access. No item-level C2/E27 image was confirmed that
systematically supersedes the DTA images. It remains a useful secondary venue
for historical expedition photographs.

### 7.2 Historical photo archives

The 2012 catalogue points to:

- F. C. Andreas photographs at Göttingen University Library;
- F. W. K. Müller/Wolfgang Lentz photographs and transcripts at the
  Asien-Afrika-Institut, Universität Hamburg;
- Olaf Hansen photographs/transcripts at the Staatsbibliothek zu Berlin.

No public item-level download portal was confirmed for these C2 collections.
Access may require direct archival inquiry.

### 7.3 IDP

[IDP's Germany collection overview](https://idp.bl.uk/blog/idp-collections-in-germany/)
states that Berlin Middle Iranian fragment images were made available through
the DTA and were not necessarily all integrated into IDP's unified holdings.
No stable IDP item page with a better C2 image was confirmed.

### 7.4 Staatsbibliothek and Museum portals

Searched:

- [Staatsbibliothek digital collections](https://digital.staatsbibliothek-berlin.de/?lang=en)
- [Museum für Asiatische Kunst Turfan research](https://www.smb.museum/en/museums-institutions/museum-fuer-asiatische-kunst/collection-research/research/turfan-expedition/)
- [SMB collections search](https://search.smb.museum/en?sammlungen=Museum+f%C3%BCr+Asiatische+Kunst)

No confirmed public record with better C2 downloads was found for tested
`MIK III` and excavation identifiers.

### 7.5 General repositories

No independent C2/E27 image corpus or machine-learning dataset was found on
Zenodo, GitHub, Figshare, or similar repositories. Online copies of
Sims-Williams 1985 reproduce the publication and plates, not a new set of
individual source photographs.

## 8. Rights and access limitations

The [DTA use rules](https://turfan.bbaw.de/dta/dta_RulesEngl_index.html) require
advance contact for publication/edition/reproduction and special permission
for commercial use. The site provides the legacy contact
`orientabt@sbb.spk-berlin.de`; verify the current address and terms before bulk
use or redistribution.

The requested citation identifies the object as a deposit of the
Berlin-Brandenburg Academy of Sciences in the Staatsbibliothek zu Berlin,
Orientabteilung, together with the exact shelfmark.

Model training, internal research use, publication of derived crops, and
redistribution of a training dataset are legally and operationally distinct.
Permission should explicitly cover the intended actions.

## 9. Recommended image-acquisition follow-up

1. Build the full C2 → E27 → `n` → `MIK III` concordance.
2. Select five representative surviving units.
3. Compare public DTA images against local plate crops at glyph level.
4. Contact BBAW/Staatsbibliothek requesting archival TIFF or highest-resolution
   JPEG access for non-commercial HTR research.
5. Ask whether the DTA's larger archival derivatives may be used to train a
   model and whether line crops/checksums may be redistributed.
6. Contact Göttingen, Hamburg, or Staatsbibliothek photo archives for lost or
   historically altered units only after the concordance identifies exact
   targets.
7. Keep multiple photographs of the same folio in one split group to prevent
   train/test leakage.

## 10. Confidence and unresolved points

| Finding | Confidence | Remaining uncertainty |
|---|---|---|
| DTA is the main public source for surviving C2/E27 fragments | High | Old interface may not expose every archival derivative |
| Public files are individual-fragment/folio images, not plate crops | High | Some DTA records may show reconstructed combinations |
| Public DTA resolution is often modest | High for sampled files | Other units may expose larger files |
| Higher-resolution institutional files likely exist | Medium-high | Availability and current permission require direct confirmation |
| No separate complete public C2 ML dataset exists | Medium-high | An unindexed/private deposit may exist |
| BLLA does not encode logical row grouping across holes | High | Future model/plugin behavior may change |
| Orli may improve line detection/order | High | Performance on C2 holes is unmeasured |
| Orli solves logical row/gap semantics | Low; no supporting evidence | Must be supplied by msocr regardless |

## 11. Final practical conclusion

The absence of local individual-fragment photographs is not absolute: many
surviving C2 pieces can be sourced individually from the DTA. Those public
images solve artificial plate-layout contamination and improve provenance, but
they do not automatically solve resolution, damage, or line grouping.

The training strategy should therefore remain source-agnostic:

1. choose the best image per folio;
2. isolate and normalize each physical fragment;
3. annotate logical rows for human context;
4. split destructive gaps into atomic readable line fragments;
5. train Kraken recognition on those atomic samples;
6. benchmark BLLA and Orli separately for layout;
7. reconstruct rows and explicit gaps in msocr, outside the recognizer.

## 12. Local E27 acquisition and preprocessing validation (2026-08-02)

This section records measured implementation results, not documentation-only
claims.

### 12.1 Reproducible local corpus acquisition

The confirmed C2/E27 concordance produced:

- 113 catalogue units/subunits;
- 103 surviving units and 10 lost units;
- 60 DTA image sets;
- 120 recto/verso JPEGs;
- 8,627,812 downloaded bytes;
- SHA-256, byte count, dimensions, source URL, shelfmark, E27 concordance, and
  rights metadata for every local image.

The JPEGs are intentionally gitignored. The tracked source manifest is
`data/manifests/c2-e27-dta-sources-v1.json`. Local integrity can be checked
without downloading again:

```bash
uv run msocr download-e27-images --verify-only
```

### 12.2 Ink-component fragment isolation is wrong for mounted DTA images

The old Sauvola + connected-component + DBSCAN path was tested on
`n484rectototal.jpg`. It returned one bounding box covering the complete
597×792 photograph, including the frame, ruler, labels, and mount. The reason
is structural: the algorithm clusters ink and other dark components, while the
desired object is the physical parchment. Frame marks, labels, manuscript ink,
and shadows form a connected spatial chain.

Therefore the DTA path now uses a separate mounted-photo procedure:

1. detect and remove dense black frame bands;
2. seed GrabCut from parchment/ink color;
3. reject components connected to the remaining photo boundary;
4. choose the large component nearest the image center;
5. whiten everything outside the selected physical-fragment mask;
6. crop and deskew the fragment;
7. recompute the geometry mask after deskew;
8. require human review of the isolation overlay.

Pilot overlays/crops were checked for `n012rectototal`, `n020rectototal`, and
`n484rectototal`. The procedure excluded external frames, rulers, and labels.
Remaining limitations were visible and important:

- `n012` retains a mounting/repair strip attached to the fragment;
- `n020` retains a modern stamp printed on the manuscript itself;
- `n484` retains the `T II B 31` mark on the manuscript and has severe fading
  and a large fissure;
- automatic masking cannot distinguish modern marks printed directly on the
  object from historical ink.

Those areas must be marked `DigitizationArtefactZone` or excluded at line level
during human annotation. They must not be silently cleaned from the pixels.

### 12.3 Default BLLA failed the first line-proposal pilot

Default Kraken BLLA was run on the masked, cropped, deskewed
`n020rectototal` fragment. It returned eight proposals. Contact-sheet review
found:

- zero complete usable text-line crops;
- five clear false positives containing parchment texture/discoloration;
- three small ink/character fragments rather than complete lines;
- clipping in two of the three ink-bearing crops.

This confirms the documented domain mismatch on an actual downloaded E27
image. Default BLLA output must **not** initialize training ground truth
automatically. `prepare-fragment-page` consequently defaults to manual
line-level annotation; `--propose-lines` is opt-in and every generated record
is marked `proposal_only`, `training_eligible: false`, and
`review_required: true`.

```bash
# Recommended first pass: physical crop/mask/deskew only
uv run msocr prepare-fragment-page \
  dataset/christian_sogdian_c2av/dta_e27/images/n020rectototal.jpg

# Benchmark/research only: emit default-BLLA proposals and a contact sheet
uv run msocr prepare-fragment-page \
  dataset/christian_sogdian_c2av/dta_e27/images/n020rectototal.jpg \
  --propose-lines
```

The practical starting point for C2 is therefore human-drawn **atomic visible
line fragments** on the isolated crop. Fine-tuned BLLA and Orli remain later
benchmarks after enough reviewed split-baseline PAGE XML exists.
