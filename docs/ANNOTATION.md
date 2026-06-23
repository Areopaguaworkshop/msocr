# How to annotate a Sogdian manuscript page

This guide walks through annotating one manuscript page in the msocr annotation
editor (`http://localhost:5173`). The output is a PAGE XML file you can feed to
Kraken `ketos train` to fine-tune the Sogdian HTR model.

## What you are doing

You are producing two kinds of marks on each page image:

- **Regions** — polygons that enclose a logical area of the page (main text
  column, margin, damage, illustration, etc.).
- **Baselines** — the reading line under each line of text. Each baseline
  carries a transcription of the Sogdian text on that line.

Kraken trains on baselines + transcriptions, not on bounding boxes. Regions are
used for layout and to exclude non-text areas from recognition.

## 1. Open a session

Start the servers:

```bash
uv run msocr annotation-api --host 127.0.0.1 --port 8001
# in another shell, the Vite dev server for hot reload:
cd frontend && npm run dev
```

Open `http://localhost:5173`. You will see existing sessions and a
"New session" form. To create one:

- **Language**: `sogdian`
- **Script variant**: `manuscript` (default)
- **Source**: a short label for the manuscript (e.g. `SO14082r`)
- **Ingestion path**: `local_file`
- **Image path**: absolute path to a page PNG on this machine
- **Crop manuscript area**: leave unchecked unless the scan has a ruler or
  frame you want to crop out.

Submit. You will be redirected to the editor at `/ui/<session_id>`.

On first load the editor calls `/api/sessions/<id>/autosuggest`, which runs
Kraken BLLA segmentation. After a few seconds you will see proposed regions
and baselines overlaid on the image. Your job is to review and fix them, then
transcribe.

## 2. Regions (press `R`)

Regions are polygons. To draw one:

1. Press `R` (or click the region icon in the left rail).
2. Pick the region type from the palette that appears top-left of the image.
   Hover any label for a 3-second tooltip explaining what it means.
3. Click points around the area. A dashed line shows the draft.
4. Double-click to close the polygon.

To edit an existing region, switch to `V` (navigate), click the region to
select it (it will be highlighted), then press `Del` to delete it and redraw.
For fine edits you can drag individual vertices — but the lazy path is to
delete and redraw, which is faster for manuscript work.

### Region types

| Label | Full name | When to use |
|---|---|---|
| Main | MainZone | The primary text column. Almost every page has one. |
| MainText | MarginTextZone | Marginal notes, glosses, or commentary around the main text. |
| Numbering | NumberingZone | Folio / page / quire numbers, signatures. |
| Damage | DamageZone | Physically damaged or unreadable area. Kraken will not try to read it. |
| Graphic | GraphicZone | Illustrations, decorations, ornaments. |
| DigitizationArtefact | DigitizationArtefactZone | Scan bleed-through, shadows, ruler marks — anything from the scanning process, not the manuscript. |
| Custom | CustomZone | Anything that does not fit the above. |

## 3. Baselines (press `B`)

A baseline is the reading line under a line of text — Kraken reads the text
that sits *above* the baseline. To draw one:

1. Press `B` (or click the baseline icon in the left rail).
2. Pick the line type from the palette top-left (hover for tooltips).
3. Click the **start** of the line (left end for RTL Sogdian — the visual
   right end of the line).
4. Click the **end** of the line. The baseline is created and selected.

That's it — two clicks. If the auto-segment's baseline is already correct,
leave it. If it is slightly off, delete it (`Del`) and redraw.

### Line types

| Label | Full name | When to use |
|---|---|---|
| Default | DefaultLine | A normal line of text. Most lines. |
| Heading | HeadingLine | A heading, title, or rubric. |
| Interlinear | InterlinearLine | A smaller line squeezed between two main lines. |

## 4. Transcribe (press `T`)

1. Press `T` (or click the transcribe icon).
2. Click a baseline on the image. The viewer auto-centres on it and the
   right-hand panel activates.
3. Type the Sogdian text on that line, right-to-left.
4. Use the character palette below the textarea to insert Sogdian characters
   (U+10F30–U+10F44) with one click if you do not have a Sogdian keyboard
   layout installed.
5. Press `Enter` to save and jump to the **next** line. The current line is
   auto-saved (2-second debounce) even if you do not press Enter.

The right panel shows a list of all lines in reading order. Drag the `⠿`
handle on any line to reorder it. Reading order matters for Kraken training:
top-to-bottom is the default for Sogdian manuscript columns.

## 5. Save and export

The editor auto-saves to the server 2 seconds after you stop editing, and
again on page unload. You can also force a save with `Ctrl+S` or the **Save**
button in the top bar.

To export for Kraken training, click **PAGE XML** in the top bar. The browser
downloads a `.page.xml` file containing:

- one `<TextRegion>` per region, with `type="paragraph"` and a custom
  `structure {type:MainZone;}` attribute;
- one `<TextLine>` per baseline, with `<Baseline points="...">` and a
  `<TextEquiv><Unicode>…</Unicode></TextEquiv>` carrying your transcription.

## 6. Keyboard shortcuts

| Key | Action |
|---|---|
| `V` | Navigate mode (pan/zoom) |
| `R` | Region mode |
| `B` | Baseline mode |
| `T` | Transcribe mode |
| `Enter` (textarea) | Save + next line |
| `Ctrl+↑` / `Ctrl+↓` | Previous / next line |
| `↑` / `↓` | Previous / next line |
| `Esc` | Cancel current drawing |
| `Del` / `Backspace` | Delete selected region or line |
| `Ctrl+S` | Save now |
| `?` (top bar) | Open this guide inside the editor |

## 7. Tips

- **Trust the auto-segment, then fix.** Kraken BLLA gets ~80% of baselines
  right on a clean scan. Spend your time on the ones that are wrong, not on
  redrawing correct ones.
- **Draw baselines under the text, not through it.** Kraken reads upward from
  the baseline, so place it at the *bottom* of the line of script.
- **Transcribe what you see, not what you expect.** Damaged or ambiguous
  characters get a `�` placeholder. Do not guess.
- **Reading order = line order in the right panel.** Drag to reorder before
  exporting.
- **One session per page.** Do not pile multiple pages into one session; the
  PAGE XML export is per-page.

## 8. Troubleshooting

- **Image does not load** — check the path you gave in the new-session form
  is readable by the server process.
- **Auto-segment returns nothing** — Kraken did not find any baselines. Draw
  them manually with `B`; regions are optional for training.
- **Zoom/pan does not work** — you are in `R` or `B` mode. Press `V` to
  navigate. The SVG overlay only captures clicks in draw modes.
- **Save shows "save failed"** — the backend is not running on port 8001.
  Restart `uv run msocr annotation-api`.
- **Characters look like boxes** — your browser does not have Noto Sans
  Sogdian. The editor loads it from Google Fonts, but offline you need it
  installed locally.