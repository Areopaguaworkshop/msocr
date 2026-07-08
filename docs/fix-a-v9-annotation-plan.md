# Fix A v9: Annotation Methodology + eScriptorium-Parity Frontend for Fragmentary Christian Sogdian

> Companion to [`fix-a-v9-fragment-pipeline.md`](./fix-a-v9-fragment-pipeline.md)
> (the general pipeline plan). This document is the deep-dive on the
> annotation stage specifically: *how* to annotate for best fine-tune
> results, AND *what eScriptorium features to implement in our existing
> React+OpenSeadragon SPA* to support both the annotation methodology and the
> fragment convention in
> [`kraken-fragmentary-manuscripts.md`](./kraken-fragmentary-manuscripts.md) §4.
>
> Date: 2026-07-07. Author: orchestrator, with @librarian (claims validation
> lib-2; eScriptorium feature inventory lib-3) and @explorer (frontend map
> exp-2).
>
> **v8 → v9 change:** v8 framed our SPA as a stopgap and eScriptorium as
> "reference only." The user's direction for v9 is the opposite: **keep our
> SPA as the platform, and implement the eScriptorium feature set in it.**
> This document now contains an explicit eScriptorium feature-parity gap
> analysis (§8) and a sized work list (§9). The methodology half (§1-3, §5-7)
> is unchanged from v8 — lib-2 validated it.

---

## Status (updated 2026-07-07)

### Stage 0.4 decision: **ANNOTATE MORE**
User confirmed 2026-07-07: target **~50 plates / ~750 lines** (up from the
current 12 plates / 178 lines). This is the single highest-leverage action in
the whole plan per `fix-a-v9-fragment-pipeline.md` §0.4. The frontend work
below is sequenced to make the scholar's annotation time efficient — the
Couture 2023 caution (protocol consistency beats quantity at small scale)
applies, so the convention checklist (§3) must be followed per-plate.

### Tier 0 — DONE (committed `bd9d187`)
All 6 Tier 0 XS items shipped in `frontend/src/AnnotateEditor.tsx` + `frontend/src/types.ts`:
1. ✅ U+0323 combining underdot in `SOGDIAN_CHARS` palette
2. ✅ DamageZone promoted out of "Advanced" toggle (default-visible region palette)
3. ✅ Keyboard shortcuts: M (mask flag), L (reading-order display), T (type assign / transcribe), Ctrl+A (select all lines)
4. ✅ Image rotation control in OSD viewer (−90° / +90° / reset)
5. ✅ Per-session `readDirection` field (default `horizontal-rl`)
6. ✅ Per-line `confidence` field placeholder in `Line` type (color-coded badge; field populated by Tier 2 #17-20 bootstrap loop)

Tier 0 items #3 (RTL `<ReadingOrder>` in PAGE XML), #7 (line `type` field), and #8 (ALTO decision) from §9 below were handled as part of Stage 0.3 backend work (commit `b0fd817`): RTL `<ReadingOrder>` parse+emit done, multi-region nesting fixed, ALTO kept+fixed (caller exists at `annotation_api.py:326`). Tier 0 frontend item #7 (line `type` field UI) is deferred to Tier 1 — backend has the region-type path, but the per-line `type` dropdown is a Tier 1 S item.

### Tier 1 — NEXT (before the scholar's next annotation batch)
These are the items that make daily annotation efficient. Do these BEFORE the
scholar starts the ~38-plate batch, otherwise annotation time is wasted
fighting the UI:
- **#10** per-point delete on baselines (Ctrl+Del) — XS
- **#11** invert reading direction (I key) — S
- **#12** join lines (J key) — S
- **#13** explicit line-region link/unlink (Y/U keys) + orphan-line indicator — S
- **#14** automatic reading-order numbering + order badges + L-key toggle (L-key already done in Tier 0, needs the auto-number sort) — S
- **#15** "?" help pop-up with shortcuts cheatsheet — S
- **#16** plain-text panel (Ctrl+5) — M
- **#7** line `type` field dropdown (Correction/Main/Numbering/Signature) — S (deferred from Tier 0)

### Tier 2 — BEFORE the bootstrap loop works end-to-end
These unblock the correct-don't-transcribe loop (§5). Tier 2 #17-20 is being
partially built NOW (Stage 2.3 of the fragment pipeline, fix-4): the
`msocr dump-preds` CLI + `/bootstrap` endpoint are landing. The frontend
wiring to call `/bootstrap` and the "Transcribe" button are the remaining
Tier 2 work after fix-4 reconciles.
- **#17** `/bootstrap` endpoint — IN PROGRESS (fix-4, Stage 2.3)
- **#18** `confidence` field — DONE in Tier 0 (placeholder populated by #17)
- **#19** model picker — S (after #17 lands)
- **#20** "Transcribe" button — S (after #17, #19)
- **#21** `row_id` fragment grouping — M (fragment convention §4.3)
- **#22** RTL `<ReadingOrder>` — DONE in Stage 0.3 backend
- **#23-26** training buttons + status + my models + copy-predict-to-manual — M each (the bootstrap loop mechanics)

### Tier 3 / Tier 4 — DEFER
Versions, active learning, lasso, scissors, IIIF, PDF, document grouping —
all deferred until the ~50-plate annotation batch is done and Stage 4
recognition fine-tune has been measured. Do not build these on speculation.

---

## 1. The core methodological choice: what kind of transcription is HTR ground truth?

Digital paleography distinguishes three transcription approaches:

- **Graphetic/diplomatic**: literal graphical representation — every
  character variant kept distinct, no abbreviation expansion, nothing
  regularized.
- **Normalized**: expands abbreviations, modernizes punctuation, adds
  reading aids — closest to what a published translation reader expects,
  furthest from the actual ink on the page.
- **Graphemic**: the middle ground — insignificant allographic variation
  (minor stroke variants of what's structurally the same letter) is merged
  into single classes, but genuine manuscript-level phenomena (diacritics,
  abbreviations, orthographic peculiarities) are preserved rather than
  silently normalized away.

### Evidence (lib-2)

- **CATMuS** (Clérice, Pinche et al., ICDAR 2024, Latin-script, 8th–16th c.
  European MSS) explicitly adopts graphemic transcription as a
  standardization choice for cross-project consistency — "a many-to-one
  mapping where allographs are normalized." It does *not* run a controlled
  experiment comparing graphemic vs diplomatic vs normalized on model
  generalization. The plan's earlier framing that CATMuS "found graphemic
  transcription produces models that generalize better than either extreme"
  **overstates the evidence**.
- **Couture et al. 2023** (JDMDH, 17th-c. French handwriting, ~1500 pages)
  provides the strongest direct evidence: their team initially used
  graphetic transcription, found it "would only create confusion for the
  algorithm," and switched to graphemic, which improved model performance.
  Crucially, they also found that *inconsistent* paleographic practices
  between transcribers caused their first model to fail — protocol
  consistency matters more than the specific protocol at small scale.
- **No study tests graphemic vs diplomatic vs normalized for Syriac-tradition
  or Sogdian scripts.** The principle is well-motivated; the empirical claim
  is extrapolation.

### What this means for us

Our existing g/γ→GAMAL collapse and t/θ qushshaya/rukkakha distinction are
well-motivated graphemic choices — merge non-distinctive variation, preserve
genuine distinctions. Keep them. The action item is **write this down as an
explicit convention document** and apply it identically across every plate,
present and future. Inconsistent graphemic judgment calls between plates (or
between annotation sessions) is a silent way to inject noise into a 178-line
dataset that's already thin — this is the Couture 2023 lesson applied to us.

One caveat from Kiessling (kraken author): graphemic/diplomatic
transcriptions are the most accurate representation of the physical text but
create a "usability gap" with tools expecting normalized text. If any
downstream use (lexicon lookup, dictionary cross-referencing, publication)
needs normalized forms, that should be a **separate post-processing/editorial
layer**, not baked into the HTR ground truth itself. Keep the training data
as close to the ink as the graphemic convention allows.

## 2. Handling damage, illegibility, and lacunae — the fragment-specific piece

This is where fragment annotation differs most from annotating a complete,
well-preserved codex, and where getting it wrong actively teaches the model
bad habits.

### The core principle
**Never transcribe what you can't actually see, even when you're scholarly
confident about the reconstruction.** If HTR ground truth contains
reconstructed readings presented as directly-read text, the model learns to
hallucinate confidently on damage — it can't tell the difference between "I
read this letter" and "the editor supplied this letter," because the ground
truth doesn't tell it. This is the single most common way editorial expertise
accidentally corrupts training data quality.

### Evidence (lib-2)

This argument is well-supported by three independent lines:

- **Mind the Gap** (Borkar & Smith, 2024, arXiv 2407.00250) explicitly tested
  training TrCroT to emit lacuna markers and achieved only 65% restoration
  accuracy — the model fails 35% of the time, directly supporting the claim
  that "reconstructed readings in HTR GT teach the model to hallucinate on
  damage."
- **Couture et al. 2023** documents that Transkribus's `<unclear>` tag causes
  the *entire line* to be excluded from training — platform-level
  enforcement of "don't train on uncertain text."
- **MAAT corpus** (Fitzgerald & Barney, 2024, ML4AL) strips Leiden conventions
  to a minimal ML format, treating restored (bracketed) text differently
  from preserved text — exactly the "separate layer" approach.

### The Leiden Conventions connection

This is the problem the Leiden Conventions were built for a century ago —
square brackets for lost text, sublinear dots for damaged/uncertain
readings, parentheses for silently-expanded abbreviations. As a Syriac
Studies scholar you already think this way for critical editions; the change
is applying the same discipline specifically to the **HTR training layer**,
kept separate from any critical-edition apparatus produced for publication.

### Two concrete options, and which one fits the current stage

1. **Simplest (recommended for now): exclude illegible spans from training
   lines entirely.** If a word or several characters are genuinely
   unreadable, don't transcribe them and don't include that line/region in
   training. Standard practice per Tarride et al. 2023 (ACM DocEng,
   crowdsourced HTR guidelines: "If several words are illegible, the
   transcription should be left blank"; "If the quality of the page is poor
   and illegible, it should be skipped") and Transkribus platform behavior.

   **Cost at our scale:** the Couture 2023 finding that `<unclear>` /
   uncertain-marked text excludes the *whole line* has a real cost at 178
   lines. Marking even one uncertain character in a line removes 5–10 words
   of training data. Reserve line-exclusion for genuinely illegible spans;
   for partially-legible single glyphs, use Leiden underdot (§3) rather
   than excluding the whole line.

2. **More sophisticated (worth revisiting with more data): explicitly train
   the model to emit lacuna markers.** Mind the Gap shows this is *possible*
   (65% > 5% baseline) but not reliable enough for production. It adds token
   complexity to a recognizer already overfitting on 178 lines (v4's
   finding). This is a "later" item, not a "now" item.

### Fragment-boundary and disconnected-piece policy
Annotate torn/damaged line-endings consistently — decide and document whether
a truncated line at a fragment's torn edge means "this line continues
off-fragment" (exclude the partial line from recognition training, since the
visible portion alone doesn't represent a complete line) versus "this is
where the line actually ends" (a real short line, include it). Do the same
for physically disconnected fragment pieces on one plate — annotate each as
its own region with an explicit reading-order position rather than letting
default reading-order inference guess at the relationship between
disconnected pieces.

## 3. Concrete line-level annotation policy (write this as a checklist)

Drawing on Tarride et al. 2023 (crowdsourced HTR guidelines), adapted for
our material:

- Only transcribe lines with clean, unambiguous segmentation. If a
  baseline/region boundary is wrong or ambiguous, fix the segmentation first
  (or exclude the line) rather than transcribing around a bad boundary.
- If a word or span is illegible, leave it out per §2 rather than guessing —
  this applies even when confident from parallel/comparable Sogdian text
  what the word likely is; that confidence belongs in the scholarly edition,
  not the HTR ground truth.
- Mark genuinely uncertain-but-attempted readings distinctly from confident
  readings using **Leiden underdot** (`U+0323` COMBINING DOT BELOW, e.g.
  `ạ`) — partially legible glyph, ink faint but trace visible. Reserve
  `�` (U+FFFD) for "there was a glyph here, I have no idea what." This
  matches `kraken-fragmentary-manuscripts.md` §3.1.
- Keep the graphemic convention from §1 identical across every line — write
  it down once, apply it by checklist, don't re-derive it per plate from
  memory. (Couture 2023: protocol consistency > quantity at small scale.)
- Skip whole lines/regions where image quality is too poor to transcribe
  confidently at all, rather than forcing a low-confidence transcription
  into the training set.

## 4. The annotation tool: our SPA, target = eScriptorium feature parity

We have a working custom annotation frontend (`frontend/`, React 18 +
TypeScript + Vite + OpenSeadragon + raw SVG overlay) backed by
`msocr/service/annotation_api.py` (FastAPI) and `msocr/data/session_manager.py`.
The v9 goal is **bring this SPA to feature parity with eScriptorium's HTR
workflow**, not migrate to eScriptorium.

Why not migrate to eScriptorium:
- We'd lose the c2av plate-gallery integration (`PlateGallery.tsx`,
  `SessionList.tsx`) that ties sessions to our specific manuscript corpus.
- We'd re-annotate from scratch — eScriptorium's import would bring PAGE XML
  over, but our session/state model doesn't map 1:1 to eScriptorium's
  document/document-part/transcription-version model.
- The pipeline plan (LOOCV, BLLA fine-tune, recognition fine-tune,
  post-correction) is wired to our `msocr` CLI and our annotation API; a
  tool switch would break that wiring.

Why target eScriptorium parity specifically (not Transkribus or a from-scratch
design):
- eScriptorium is the reference open-source HTR annotation platform, wraps
  Kraken (our engine), and its workflow is documented well enough to copy
  feature-by-feature ([escriptorium.readthedocs.io](https://escriptorium.readthedocs.io/en/latest/),
  [UB Mannheim training guide](https://ub-mannheim.github.io/eScriptorium_Dokumentation/Training-with-eScriptorium-EN.html)).
- Transkribus is proprietary/cloud; we can't self-host or copy its
  features.
- A from-scratch design risks reinventing eScriptorium badly. Copying a
  known-good feature set is cheaper and lower-risk than designing under
  uncertainty.

§8 contains the full feature-parity gap analysis (lib-3 enumerated
eScriptorium's features; exp-2 mapped our SPA's current state). §9 is the
sized, ordered work list. §10 is what we explicitly skip and why.

## 5. The bootstrap workflow: correct, don't transcribe from scratch

This is the single highest-leverage annotation-efficiency change available,
and it's eScriptorium's own documented intended workflow (feature 5.4 in
lib-3's inventory): manually segment/transcribe an initial subset, train a
model on it, run that model on the next batch of unannotated material, and
have a human **correct** the model's output rather than transcribe blind.

The 15-corrected-pages claim (German early-print, pseudo-Vincent Ferrer
*De fine mundi*, Augsburg 1486, traced via Stefan Weil / Jonathan Green's
2023 blog) is real and documented. **Caveat for our domain:** that case was
early German *print* (Fraktur) on top of a strong `german_print` base model.
Our material is handwritten, cursive, fragmentary, in a script with no
public base model of comparable quality. The number of pages needed for a
usable work-specific model on Christian Sogdian is likely higher than 15.

### Critical caution: round-1 may give NO speedup

**Mohammad et al. 2025 (ACM DocEng)** instrumented eScriptorium with a
tracing layer and measured actual transcription time across three
workflows. Key finding: **a default (non-fine-tuned) model gave about the
same time as fully manual transcription, mainly due to the long correction
phases. Only a fine-tuned model was meaningfully faster.**

Our v2 shipped model (CER 77.75%) is effectively a default model for new
c2av plates. Expect the first round of correct-don't-transcribe to be
~as slow as transcribing from scratch. The speedup (if any) comes in round
2+ *after* the pipeline plan's Stage 4 produces a better fine-tuned
recognizer. **Do not budget scholar-hours assuming round-1 speedup.**

### Current state of our `/autosuggest` (this becomes the "Transcribe" button in eScriptorium terms)

The endpoint exists and the frontend calls it on mount when saved
annotations are empty. It runs Kraken BLLA with
`text_direction="horizontal-rl"` and returns `{regions, lines}` in v2
annotation shape. **But it returns geometry only (baselines + boundaries),
no recognition predictions and no per-line confidence scores.** This maps
to eScriptorium feature 3.6 (auto-segmentation) but not 4.6
(auto-transcription). To match eScriptorium, we need a separate
"Transcribe" action that runs the Kraken recognizer on already-segmented
lines and fills the `transcript` field, plus a per-line `confidence` field
that the UI displays. See §9 Tier 2.

### Concrete workflow for the next batch of c2av (or other Turfan) plates

1. Run "Segment" (current `/autosuggest`) over the new plate to get
   baselines + boundary polygons. (eScriptorium feature 3.6.)
2. Correct segmentation: delete wrong lines, add missed lines, fix
   boundaries, set reading order, assign region types including DamageZone.
   (eScriptorium features 3.2-3.5.)
3. Run "Transcribe" (new endpoint, §9 Tier 2) to pre-fill transcripts with
   per-line confidence. (eScriptorium feature 4.6.)
4. Correct rather than retype — fix wrong characters, mark illegible spans
   per §2, don't retranscribe lines the model already got right. Use the
   virtual keyboard (character palette) for rare glyphs. (eScriptorium
   features 4.1, 4.4.)
5. After a fine-tuned model exists (pipeline plan Stage 4), the loop gets
   faster. Don't expect round-1 speedup (Mohammad 2025).

## 6. Which lines to prioritize annotating next (active learning, not sequential)

Don't annotate the next plate front-to-back by default. Two prioritization
strategies, usable together:

- **Rare-class targeting** (carried from v5/v6): prioritize plates/lines
  likely to contain qushshaya, rukkakha, zhain, khaph, fe — the rarest
  classes by a wide margin, and the ones with no public-dataset substitute.

- **Confidence-based active learning**: run the current model over
  candidate unannotated lines and rank them by the model's own per-line
  confidence score, lowest first. The principle is established (Tsakalis
  2021 Cambridge MPhil thesis found up to 18% annotation cost savings and
  ~4.4pp WER improvement on IAM; VUT Brno 2024/2025 found confidence-based
  selection cut annotation in half starting from just 16 lines).

### Caveat for our scale (lib-2)

**Both studies operate on datasets 10–100× larger than 178 lines.** At our
scale, confidence estimates may be too poorly calibrated for uncertainty
sampling to help — a known failure mode is that overconfident wrong
predictions on rare classes get de-prioritized. The combination strategy
(rare-class targeting first, then within those rank by lowest confidence)
is the safer ordering: rare-class targeting doesn't depend on confidence
calibration. Treat pure confidence-based active learning as a "later"
lever once the dataset is larger and confidence estimates are better
calibrated. **This feature depends on §9 Tier 2 #7 (per-line confidence
field) landing first.**

## 7. Ground truth reuse and sharing

Two directions, quickly:

- **Check whether comparable ground truth already exists** before annotating
  more from scratch. The HTR United catalogue
  ([htr-united.github.io](https://htr-united.github.io/catalog.html))
  aggregates publicly available HTR ground truth across projects. Search
  done (lib-2): two Syriac datasets from the Vienna HTR Winter Schools are
  catalogued — ÖNB Cod. Syr. 1 (Serto, 2869 lines, CC-BY 4.0,
  [Zenodo 14714089](https://doi.org/10.5281/zenodo.14714089)) and MS
  Jerusalem Saint Mark's Monastery 36 (Estrangelo, 12th–14th c.,
  [Zenodo 18157525](https://doi.org/10.5281/zenodo.18157525)). **No
  Sogdian, Christian Sogdian, Manichaean, or Eastern Syriac HTR ground
  truth was found in HTR United or Zenodo.** The Winter School Syriac
  guidelines explicitly *exclude* qushoyo and rukokho (the Syriac analogues
  of our qushshaya/rukkakha) — so even the closest public dataset cannot
  help with our rarest classes. We cannot offload annotation effort to
  existing datasets.

- **Consider publishing our c2av ground truth once it's in reasonable
  shape**, with clear attribution/citation metadata. There's active
  methodological work on best practices for sharing and reusing ground
  truth in HTR infrastructures, aimed at exactly this kind of low-resource,
  high-scholarly-value material. This would sit naturally alongside
  GCDFL's existing open-scholarship mission, and means future work on
  Christian Sogdian or related Eastern-Syriac-tradition material doesn't
  start from zero the way this project did.

## 8. eScriptorium feature-parity gap analysis (lib-3 vs exp-2)

eScriptorium has ~120 documented features (lib-3). Not all make sense for a
single-annotator Syriac-project SPA. This section maps each eScriptorium
feature group onto our SPA's current state, marks gaps, and sizes the work.
§10 documents the explicit skips.

Legend: **have** = works now; **partial** = exists but incomplete; **gap**
= not implemented; **skip** = deliberately not implementing (see §10).

### 8.1 Document/project management (eScriptorium §1)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Projects (top-level container) | **skip** — one project, the c2av corpus | — | — |
| Documents (logical units) | **partial** — sessions are per-plate, no document grouping | Add document grouping if we ever annotate >1 manuscript | L (defer) |
| Document-parts (individual images) | **have** — one session = one plate | — | — |
| Document metadata (name, script, read direction, line offset) | **partial** — session has plate_path, no script/direction/offset fields | Add metadata fields to session model | S |
| Read direction setting (LTR/RTL) | **have** — hardcoded RTL for Sogdian; `--text-direction` on CLI | Surface in UI, make per-session | XS |
| Line offset (baseline/topline) | **gap** — we assume baseline | Add field, default baseline | XS |
| Confidence visualization toggle | **gap** — no confidence field yet (see §8.4) | Depends on §9 Tier 2 #7 | S |
| Tags on documents | **skip** — single-annotator, single-corpus | — | — |
| Tag management, filtering | **skip** | — | — |
| Project/document/model sharing, teams, invitations | **skip** — single-user | — | — |
| User profile, API key, onboarding, password mgmt | **skip** — no auth (local tool) | — | — |
| Document rename/delete | **have** — session list supports delete | — | — |

### 8.2 Image import & preprocessing (eScriptorium §2)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Local file upload | **have** — c2av plate gallery + arbitrary image path | — | — |
| PDF import | **have** — via `msocr htr` CLI, not in SPA | Add to SPA if needed | M (defer) |
| IIIF import | **gap** — no IIIF client | Add if we ingest from IIIF-hosted MSS | M (defer) |
| Image rotation | **gap** — OSD supports it, no UI control | Add rotation control to OSD viewer | XS |
| Binarization | **have** — `msocr preprocess` CLI; eScriptorium docs say "not recommended" | — | — |
| Image ordering, thumbnail dashboard, shift-click interval | **have** — `PlateGallery.tsx` + `SessionList.tsx` | — | — |

### 8.3 Segmentation (eScriptorium §3)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Ontology panel (region/line types) | **partial** — region types in `REGION_TYPES`, no UI editor | Add editable ontology if we add custom types | S (defer) |
| Default region types (Commentary, Illustration, Main, Title) | **have** — 7 SegmOnto types including Main, Margin, Damage | — | — |
| Default line types (Correction, Main, Numbering, Signature) | **gap** — lines are untyped | Add line `type` field | S |
| Custom region/line types | **gap** — fixed `REGION_TYPES` | Add editable list | S (defer) |
| SegmOnto compatibility | **have** | — | — |
| Type assignment (T key, NumPad) | **partial** — region type dropdown exists; no keyboard shortcut | Add T-key shortcut | XS |
| Region color customization | **have** — per-type colors | — | — |
| Region mode toggle (R key) | **have** — region/line drawing toggle | — | — |
| Draw rectangular regions | **have** — polygon regions (more flexible than rect) | — | — |
| Modify region polygon (drag vertices, add vertex) | **have** | — | — |
| Delete region (Del key) | **have** | — | — |
| Nested/overlapping regions | **partial** — possible, same line-region caveat as eScriptorium | — | — |
| Draw baselines (2-point) | **have** | — | — |
| Add points to baseline (double-click) | **have** | — | — |
| Move baseline points (drag) | **have** | — | — |
| Delete point from baseline (Ctrl+Del) | **gap** — no per-point delete | Add | XS |
| Delete entire line (Del) | **have** | — | — |
| Masks (polygons around baselines) | **have** — Kraken-computed, shown in SVG | — | — |
| Mask display toggle (M key) | **partial** — visible, no toggle shortcut | Add M-key | XS |
| Manual mask calc trigger | **have** — `/autosuggest` returns masks | — | — |
| Mask recalc on baseline edit | **gap** — masks not auto-updated on edit | Re-run Kraken bbox on edit (or defer to next Segment) | M (defer) |
| "Only line Masks" segmentation | **gap** — no separate mask-only action | Add if needed | S (defer) |
| Invert reading direction (I key) | **gap** — no per-line direction invert | Add I-key | S |
| Join lines (J key) | **gap** — no join | Add J-key | S |
| Scissors tool (C key) | **gap** — no scissors | Add C-key | M (defer) |
| Lasso selection (Shift+drag) | **gap** — single selection only | Add lasso | M |
| Select all (Ctrl+A) | **gap** | Add Ctrl+A | XS |
| Shift+click add/remove from selection | **gap** | Add | S |
| Move selection (Ctrl+drag) | **gap** — moves one at a time | Add | S |
| Undo/Redo (Ctrl+Z / Ctrl+Y) | **gap** — no undo stack | Add undo/redo | M |
| Cancel drawing (Esc) | **have** | — | — |
| Automatic reading order | **gap** — array order only, no auto-sort | Add top-to-bottom auto-sort | S |
| Ordering display toggle (L key) | **gap** — no order numbers shown | Add L-key + render order badges | XS |
| Manual reorder via Text panel | **partial** — drag-to-reorder in line list | — | — |
| Order recalc on edit | **gap** — array order is manual | Auto-renumber on add/delete | XS |
| Link lines to region (Y key) | **partial** — implicit by geometry, no explicit link | Add explicit Y/U link/unlink | S |
| Unlink lines (U key) | **gap** | Add with Y | S |
| Orphan lines, empty regions | **partial** — line can exist without region | — | — |
| Segment button (bulk apply seg model) | **have** — `/autosuggest` per session | Bulk across sessions? Defer | L (defer) |
| Segmentation levels (4 options) | **partial** — only "Lines Baselines and Masks" | Add level select if needed | S (defer) |
| Reading direction for prediction | **have** — hardcoded horizontal-rl | Surface as option | XS |
| Override checkbox | **have** — autosuggest replaces | — | — |
| Per-image cancel, done notification | **skip** — synchronous, one plate at a time | — | — |
| "?" help pop-up | **gap** — no in-app help | Add shortcuts cheatsheet | S |

### 8.4 Transcription (eScriptorium §4)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Transcription panel (Ctrl+4) — overlay text on image | **have** — inline text editor per line | — | — |
| Text panel (Ctrl+5) — plain-text block view | **gap** — no plain-text panel | Add plain-text panel | M |
| Line navigation (Enter/↓/↑) | **have** — Tab/Enter navigation | — | — |
| Line-image highlight on hover | **gap** — no cross-panel hover | Add | S |
| Auto-save | **have** — autosave on blur | — | — |
| Multiple transcription versions | **gap** — one transcript per line | Add version field, named versions | L |
| "manual" version default | **have** — implicit | — | — |
| Model-named versions (kraken:{model}) | **gap** — predict overwrites | Add version on predict | M |
| Import-named versions | **gap** — import overwrites | Add version on import | S |
| Version switcher dropdown | **gap** | Add | S |
| Delete transcription version | **gap** | Add | S |
| Line-level history (per-line edit log) | **gap** — no history | Add history table | M (defer) |
| Transcription comparison (diff) | **gap** | Add diff view | M (defer) |
| Virtual keyboard toggle | **have** — character palette | — | — |
| Create custom keyboard | **gap** — fixed palette | Add editable keyboard JSON | M (defer) |
| Import keyboard (JSON) | **gap** | Add | S (defer) |
| Export keyboard (JSON) | **gap** | Add | XS (defer) |
| Multi-character keys | **gap** — single chars only | Add | S (defer) |
| Keyboard manager | **gap** | Add | S (defer) |
| Per-document, per-user keyboards | **skip** — single user | — | — |
| Text annotation categories (ontology) | **gap** — no text-level annotation | Add | M (defer) |
| Apply text annotation | **gap** | Add with ontology | M (defer) |
| Remove text annotation | **gap** | Add | XS (defer) |
| Transcribe button (bulk apply recognition model) | **gap** — `/autosuggest` is geometry-only | Add recognition to autosuggest or new endpoint | M |
| Model selection for transcription | **gap** — one model, env-resolved | Add model picker | S |
| New transcription version on predict | **gap** | Add with versions | M |
| Per-image cancel, done notification | **skip** — synchronous | — | — |

### 8.5 Training / bootstrap loop (eScriptorium §5)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Train Recognizer (UI) | **gap** — `msocr train` CLI only | Add "Train" button that calls CLI | M |
| Train Segmenter (UI) | **gap** — `msocr train` CLI only | Add "Train Segmenter" button | M |
| Training rights | **skip** — single user | — | — |
| Training from scratch | **have** — via CLI | Wire to UI | — |
| Fine-tuning (select base model) | **have** — via CLI `--base-model` | Add base-model picker in UI | S |
| Transcription version selection for training | **gap** — uses whatever's in session | Tie to versions (after §8.4 versions) | M |
| Model naming | **have** — `--name` CLI flag | Add name field in UI | XS |
| Overwrite checkbox | **have** — CLI flag | — | — |
| All training data in one document | **skip** — our model is per-session, multi-plate via manifest | — | — |
| Training status display | **gap** — no UI status | Poll task, show "training"/"done" | S |
| Toggle Versions (epochs) | **gap** — no epoch browser | List epochs, download .mlmodel per epoch | M (defer) |
| Best model selection | **have** — Kraken picks | — | — |
| Task reports / monitoring | **gap** — CLI prints, no UI | Add task log view | S (defer) |
| Training notification | **gap** | Add toast on completion | XS |
| My Models page | **gap** — models dir only | List models in `models/kraken/` | S |
| Upload a model | **gap** — filesystem only | Add upload | S (defer) |
| Download a model | **have** — filesystem | — | — |
| Model roles (Segment/Recognize) | **partial** — implicit by file | Add role tag | XS |
| Document Models tab | **skip** — per-session | — | — |
| Copy-predict-to-manual workflow | **gap** — predict overwrites manual | Add "copy to manual" action | S |
| Iterative fine-tuning loop | **have** — manual via CLI | Make it one-click from UI | M |
| Segmentation correction before transcription | **have** — workflow order | — | — |
| Segmentation fine-tuning loop | **have** — via CLI | — | — |
| Model evaluation via comparison | **gap** — no side-by-side | Add compare view | M (defer) |

### 8.6 Export (eScriptorium §6)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Export button | **have** — PAGE XML download | — | — |
| Transcription version selection | **gap** — exports current | Tie to versions | M |
| PAGE XML export | **have** — `_export_page_xml_v2` | — | — |
| ALTO XML export | **partial** — `_export_alto` exists but reads v1 state (exp-2) | Fix to v2 or delete | S |
| Plain text export | **gap** | Add | XS |
| Include images option | **gap** | Add zip with images | S (defer) |
| Region type filtering | **gap** — exports all | Add checkboxes | S |
| Region types in XML only | **have** | — | — |
| Empty regions in export | **have** | — | — |
| Export without segmentation | **have** | — | — |
| Export without transcription | **have** | — | — |
| Download link | **have** | — | — |
| Permanent export links | **skip** — files are local | — | — |
| METS XML manifest | **gap** — no METS | Add for multi-plate exports | M (defer) |
| Model export (download) | **have** — filesystem | — | — |

### 8.7 Collaboration / review (eScriptorium §7)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Multi-user instance | **skip** | — | — |
| Project/document/model sharing, teams | **skip** | — | — |
| Transcription history (per-line) | **gap** | Add with versions | M (defer) |
| Transcription version comparison | **gap** | Add diff view | M (defer) |
| Tags as workflow states | **skip** | — | — |
| Tag-based filtering | **skip** | — | — |
| Formal review states | **skip** | — | — |
| Comments on transcriptions | **skip** | — | — |
| Image annotations with comments | **gap** — no separate image-annotation layer | Add image-annotation layer (see §8.9) | M (defer) |
| Document versioning | **skip** — session JSON is the version | — | — |

### 8.8 Search (eScriptorium §8)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Full-text search (ElasticSearch) | **skip** — overkill for ~50 plates | — | — |
| Search scope, exact/fuzzy/phrase | **skip** | — | — |
| Search results display | **skip** | — | — |

### 8.9 Image annotations (separate from segmentation) (eScriptorium §9)

This is a distinct eScriptorium concept from segmentation regions. Image
annotations are scholarly markup on the image (damage zones, decoration,
later hands, water damage) that are NOT SegmOnto text regions. Our current
approach conflates these by putting DamageZone in the segmentation ontology
(which is SegmOnto-compliant and fine), but we have no separate layer for
non-segmentation scholarly markup.

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| Image annotation categories (ontology) | **gap** — no separate annotation layer | Add separate image-annotation layer | M (defer) |
| Draw rectangle/polygon annotation | **gap** | Add | M (defer) |
| Annotation comment dialog | **gap** | Add with layer | S (defer) |
| Delete image annotation | **gap** | Add | XS (defer) |
| Delete annotation category | **gap** | Add | XS (defer) |
| Export annotations | **gap** | Add once layer exists | S (defer) |

**Decision for v9:** DamageZone stays as a SegmOnto region type (it IS a
region type per `kraken-fragmentary-manuscripts.md` §4). The separate
image-annotation layer is deferred — it's scholarly markup, not training
data, and the c2av publication pipeline handles that elsewhere. Revisit if
annotation workflow demands it.

### 8.10 API (eScriptorium §10)

| eScriptorium feature | Our SPA state | Gap work | Size |
|---|---|---|---|
| REST API | **have** — `annotation_api.py` | — | — |
| API authentication token | **skip** — local tool | — | — |
| Python connector library | **skip** — `msocr` CLI is our connector | — | — |
| Training via API | **gap** — CLI only | Add `/train` endpoint if UI needs it | M (defer) |

### 8.11 Administration (eScriptorium §11)

All skipped — single-user local tool. No Django admin, no leaderboard, no
rights management, no ElasticSearch config.

## 9. Work list: bringing our SPA to eScriptorium parity (sized, ordered)

Ordered by (dependency, leverage, effort). XS = <1hr, S = <4hr, M = ~1 day,
L = >2 days. Ponytail: do the smallest thing that unblocks the next tier.

### Tier 0 — do first (XS, unblock the fragment convention + daily annotation)

These are tiny, high-leverage, and the user is probably already hitting
their absence.

1. **Add `U+0323` (combining underdot) to the character palette.** One
   array entry in `SOGDIAN_CHARS`. Textarea already accepts any Unicode.
   **XS.** (Leiden underdot for partial-legibility marking, §3.)
2. **Promote DamageZone out of the "Advanced" toggle** to default-visible
   in the region palette. **XS.** (Fragment convention §4 requires it.)
3. **Add `readingDirection {value:right-to-left;}` custom attribute to
   `<TextLine>` in `_export_page_xml_v2`.** Parse it back in
   `parse_page_xml_to_v2`. **S.** (`kraken-fragmentary-manuscripts.md` §4.2.)
4. **Add keyboard shortcuts: M (mask toggle), L (order display), T (type
   assign), Ctrl+A (select all).** **XS each.** (eScriptorium parity,
   §8.3.)
5. **Add image rotation control** to OSD viewer. **XS.** (eScriptorium
   §2.4.)
6. **Surface read direction as per-session field** (default
   `horizontal-rl`). **XS.** (eScriptorium §1.5.)
7. **Add line `type` field** (default Main, options per SegmOnto line
   types: Correction, Main, Numbering, Signature). **S.** (eScriptorium
   §3.1.3.)
8. **Delete or fix `_export_alto` for v2 state.** Ponytail: delete unless a
   consumer needs ALTO — PAGE XML is canonical. **XS to delete, S to fix.**
   (exp-2 finding.)

### Tier 1 — do before the next annotation batch (S, complete the segmentation feature set)

9. **Fix multi-region nesting in `_export_page_xml_v2`** — track
   line→region membership, emit each line inside its actual region.
   **S.** (Currently all lines nest under the first region — wrong for
   multi-region fragmentary folios. exp-2 finding.)
10. **Add per-point delete on baselines (Ctrl+Del).** **XS.** (eScriptorium
    §3.3.4.)
11. **Add invert reading direction (I key).** **S.** (eScriptorium §3.3.11.)
12. **Add join lines (J key).** **S.** (eScriptorium §3.3.12.)
13. **Add explicit line-region link/unlink (Y/U keys)** with visual
    indicator for orphan lines. **S.** (eScriptorium §3.5.1-2.)
14. **Add automatic reading-order numbering** (top-to-bottom sort on
    demand) + render order badges + L-key toggle. **S.** (eScriptorium
    §3.4.)
15. **Add "?" help pop-up** with shortcuts cheatsheet. **S.** (eScriptorium
    §3.7.1.)
16. **Add plain-text panel** (Ctrl+5) — all lines as a sequential editable
    text block, autosave on blur, hover-highlight cross-panel. **M.**
    (eScriptorium §4.1.2-4.)

### Tier 2 — do before the bootstrap loop works end-to-end (M, the actual HTR loop)

17. **Add recognition to `/autosuggest` (or new `/bootstrap` endpoint).**
    Run the Kraken recognizer on each segmented line, populate
    `transcript` + `confidence`. Generalize
    `scripts/dump_c2av12_preds.py` into a `msocr dump-preds` subcommand
    first (pipeline plan §2.3), then wire the frontend to call it.
    **M.** (eScriptorium §4.6.1. Unblocks §5 bootstrap, §6 active learning.)
18. **Add `confidence` field to the `Line` type** (frontend + v2 state +
    backend) and render a small confidence badge in the line list. **S.**
    (eScriptorium §1.7, §4.6. Active learning §6 needs this.)
19. **Add model picker for transcription** — list `models/kraken/*.mlmodel`,
    pass chosen path to the recognize call. **S.** (eScriptorium §4.6.2.)
20. **Add "Transcribe" button** distinct from "Segment" — runs recognition
    on existing segmentation, fills `transcript` + `confidence`. **S.**
    (After #17.)
21. **Add `row_id` (fragment grouping) to the `Line` type.** In the UI,
    add a "group fragments into row" action (auto-group by Y-proximity as
    a first pass, manual override). In export, use the `l5_r`/`l5_l`
    naming convention (MVP) or a `custom="structure {type:DefaultLine;
    row:5; fragment:r;}"` attribute. **M.** (Fragment convention §4.3.)
22. **Add RTL `<ReadingOrder>` element to `_export_page_xml_v2`.** Emit
    `<ReadingOrder><OrderedGroup>` with lines in array order. Parse it
    back. **M.** (Pairs with #14 reading-order numbering, #21 fragment
    grouping.)
23. **Add "Train Recognizer" / "Train Segmenter" buttons** that call
    `msocr train` / `msocr train --segmenter` with the current session as
    ground truth, base-model picker, name field. **M.** (eScriptorium
    §5.1.1-2.)
24. **Add training status display** — poll task, show "training"/"done",
    toast on completion. **S.** (eScriptorium §5.2.1, §5.2.6.)
25. **Add "My Models" view** — list `models/kraken/*.mlmodel` with role
    tags (Segment/Recognize) and download links. **S.** (eScriptorium
    §5.3.1.)
26. **Add "copy predict to manual" action** — copy auto-transcription into
    the manual version for correction. **S.** (eScriptorium §5.4.1. The
    core bootstrap-loop mechanic.)

### Tier 3 — do when versions/active learning land (M, defer)

27. **Add transcription versions** (manual + model-named + import-named),
    version switcher dropdown, version delete. **L.** (eScriptorium §4.2.
    The big one — enables history, comparison, safe re-transcription.)
28. **Add transcription version selection for training** — choose which
    version is ground truth. **M.** (After #27.)
29. **Add line-level history** (per-line edit log with attribution). **M.**
    (eScriptorium §4.3.1. After #27.)
30. **Add transcription comparison** (diff view between versions). **M.**
    (eScriptorium §4.3.2. After #27.)
31. **Add epoch browser** — list training epochs per model, download
    per-epoch `.mlmodel`. **M.** (eScriptorium §5.2.2.)
32. **Add model evaluation comparison** — side-by-side diff of two
    models' transcriptions. **M.** (eScriptorium §5.4.5.)
33. **Add undo/redo stack** (Ctrl+Z / Ctrl+Y). **M.** (eScriptorium
    §3.3.18. Big because it touches all drawing ops.)

### Tier 4 — do if/when scope grows (L, defer)

34. **Lasso selection (Shift+drag), multi-select move, shift+click
    add/remove.** **M.** (eScriptorium §3.3.14-17.)
35. **Scissors tool (C key).** **M.** (eScriptorium §3.3.13.)
36. **PDF import in SPA.** **M.** (eScriptorium §2.2.)
37. **IIIF import.** **M.** (eScriptorium §2.3.)
38. **Document grouping** (multiple manuscripts). **L.** (eScriptorium
    §1.2.)
39. **Bulk Segment/Transcribe across sessions.** **L.** (eScriptorium
    §3.6.1, §4.6.1.)
40. **Editable virtual keyboard** (create/import/export JSON,
    multi-char keys). **M.** (eScriptorium §4.4.)
41. **Text annotation categories** (text-level markup, separate from
    image annotations). **M.** (eScriptorium §4.5.)
42. **Separate image-annotation layer** (non-segmentation scholarly
    markup). **M.** (eScriptorium §9.)
43. **METS XML manifest for multi-plate exports.** **M.** (eScriptorium
    §6.14.)
44. **Editable ontology** (custom region/line types). **S.** (eScriptorium
    §3.1.4.)
45. **Mask recalculation on baseline edit.** **M.** (eScriptorium
    §3.3.9.)
46. **"/train" API endpoint** for programmatic training. **M.** (eScriptorium
    §10.4.)

## 10. Explicitly out of scope (ponytail: not implementing, and why)

These eScriptorium features are skipped deliberately, not forgotten:

- **Multi-user / auth / teams / sharing / invitations / API tokens /
  admin backend / leaderboard.** Single-annotator local tool. Adding auth
  is a multi-day tangent with no leverage for the actual HTR work.
- **Tags on documents, tag management, tag filtering.** Single corpus,
  single annotator — directory structure is enough.
- **Full-text search (ElasticSearch).** Overkill for ~50 plates. `grep`
  on exported PAGE XML does the job.
- **Project/document deletion constraints** (eScriptorium's "can't delete
  projects" limitation). We're not eScriptorium; we can delete sessions.
- **Per-document, per-user keyboards.** Single user. One global palette
  is fine.
- **Formal review states / comments.** Single annotator, no review
  pipeline. If peer review happens, it happens in the scholarly edition,
  not the HTR tool.
- **Document versioning / branching.** Session JSON + git is our version
  control.
- **Binarization UI button.** eScriptorium docs themselves say "not
  recommended." We have `msocr preprocess` CLI if needed.
- **eScriptorium's "training from scratch may hit memory limits" — not
  our problem, we use RunPod (pipeline plan).**
- **eScriptorium's per-image cancel / done notifications.** Synchronous
  single-plate tool; not needed.

Revisit any of these if the project grows a second annotator, a peer-review
pipeline, or a public deployment. Until then, YAGNI.

## 11. Summary checklist for the next annotation batch

1. Write down the graphemic transcription convention once (§1), apply it
   identically every session. (Couture 2023: consistency > quantity.)
2. Exclude illegible spans rather than reconstruct them (§2) — keep
   critical-edition reconstruction as a separate layer. Use Leiden underdot
   for partially-legible single glyphs, not whole-line exclusion.
3. Annotate fragment boundaries and disconnected pieces with an explicit,
   documented convention (§2). Use `row_id` for fragment grouping once §9
   Tier 2 #21 is done.
4. Use the bootstrap correct-don't-transcribe loop (§5) for every new
   plate — but don't budget round-1 speedup (Mohammad 2025). Needs §9
   Tier 2 #17-20, 26.
5. Prioritize which lines to annotate next by rare-class content first,
   then by lowest model confidence within those (§6). Pure
   confidence-based active learning is unvalidated at 178 lines — treat
   as a later lever. Needs §9 Tier 2 #18 (confidence field).
6. Check HTR United (done — no Sogdian GT exists, §7) and consider
   eventual publication of our own GT.
7. Do Tier 0 + Tier 1 frontend fixes (§9) before the next annotation batch
   — they are XS-S and unblock both the fragment convention and
   eScriptorium-parity segmentation.
8. Do Tier 2 before expecting the bootstrap loop to actually save time
   (§5, §6).

## Sources (new in v9)

- eScriptorium official docs: [escriptorium.readthedocs.io](https://escriptorium.readthedocs.io/en/latest/) (all walkthrough pages; shortcuts; API)
- UB Mannheim eScriptorium training guide: [ub-mannheim.github.io/eScriptorium_Dokumentation](https://ub-mannheim.github.io/eScriptorium_Dokumentation/Training-with-eScriptorium-EN.html)
- eScriptorium feature inventory (lib-3, this session): full enumeration by workflow stage, used as the parity reference for §8.

Sources carried from v8 (methodology half, lib-2 validated):

- CATMuS Medieval (Clérice, Pinche et al., ICDAR 2024): [hal-04453952](https://inria.hal.science/hal-04453952) · [catmus-guidelines.github.io](https://catmus-guidelines.github.io/)
- Couture et al. 2023 (HTR model training challenges, protocol consistency, `<unclear>` line-exclusion): [JDMDH 10.46298/jdmdh.10542](https://doi.org/10.46298/jdmdh.10542)
- Mohammad et al. 2025 (assisted-transcription time measurement, default-model no-speedup finding): [DocEng 10.1145/3704268.3748677](https://doi.org/10.1145/3704268.3748677)
- Tsakalis 2021 (active learning for historical HTR, IAM dataset): [Cambridge MPhil thesis](https://www.mlmi.eng.cam.ac.uk/files/konstantinos_tsakalis_8224911_assignsubmission_file_dissertation_signed.pdf)
- VUT Brno 2024/2025 (practical fine-tuning, 16-line start): [fit.vut.cz/research/result/c197674](https://www.fit.vut.cz/research/result/c197674/.en)
- Tarride et al. 2023 (crowdsourced HTR annotation guidelines, exclude-illegible standard): [DocEng 10.1145/3604951.3605517](https://doi.org/10.1145/3604951.3605517)
- Mind the Gap (Borkar & Smith, 2024, lacuna-marker training): [arXiv 2407.00250](https://arxiv.org/html/2407.00250)
- MAAT corpus (Fitzgerald & Barney, 2024, Leiden-stripped ML format): [ML4AL 10.18653/v1/2024.ml4al-1.7](https://doi.org/10.18653/v1/2024.ml4al-1.7)
- HTR United catalogue: [htr-united.github.io](https://htr-united.github.io/catalog.html)
- HTR Winter School 2024 Syriac (ÖNB Cod. Syr. 1): [Zenodo 10.5281/zenodo.14714089](https://doi.org/10.5281/zenodo.14714089)
- HTR Winter School 2025 Syriac (MS Jerusalem Saint Mark's 36): [Zenodo 10.5281/zenodo.18157525](https://doi.org/10.5281/zenodo.18157525)
- Weil / Green 2023 (15-corrected-pages early-print claim): [Research Fragments blog](http://researchfragments.blogspot.com/2023/08/escriptorium-is-bad-and-brilliant.html)

Companion document: [`fix-a-v9-fragment-pipeline.md`](./fix-a-v9-fragment-pipeline.md) — the general pipeline plan (LOOCV, preprocessing wiring, BLLA fine-tune, recognition fine-tune, post-correction, inference).