import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import {
  ArrowDown,
  ArrowUp,
  Cursor,
  Eraser,
  ScribbleLoop,
  TextAUnderline,
  Trash,
  DownloadSimple,
  FileArrowUp,
  FloppyDisk,
  Path as PathIcon,
  Question,
  X,
} from "@phosphor-icons/react";
import type {
  AnnotationState,
  Line,
  LineType,
  Mode,
  Point,
  Region,
  RegionType,
} from "./types";

// Sims-Williams Sogdian Latin transliteration (per English-index-dictionary-sogdian.pdf,
// right column). Includes aleph/ayin (U+02BE/02BF), Greek β γ δ θ for Sogdian extras,
// dotted/underdotted ṭ ṣ ẓ ḥ ḣ, caron š ž č ǰ, and long vowels ā ī ū.
// ponytail: palette mirrors the Semitic-order chart in the PDF; type freely if a char isn't here.
const SOGDIAN_CHARS = [
  "ʾ", "b", "g", "d", "h", "w", "z", "ḥ", "ṭ", "y", "k", "l", "m", "n",
  "s", "ʿ", "p", "ṣ", "q", "r", "š", "t", "θ",
  "β", "γ", "δ", "x", "č", "ž", "ẓ", "ḣ",
  "ā", "ī", "ū", "ʾ", "ʿ",
  // ponytail: U+0323 COMBINING DOT BELOW — Leiden underdot for uncertain glyphs
  // (kraken-fragmentary-manuscripts.md §4.1). Combining char; applies to preceding base.
  "◌̣",
];

const REGION_TYPES: RegionType[] = [
  "MainZone", "MarginTextZone", "NumberingZone",
  "DamageZone", "GraphicZone", "DigitizationArtefactZone", "CustomZone",
];
const LINE_TYPES: LineType[] = ["DefaultLine", "HeadingLine", "InterlinearLine"];

const REGION_COLORS: Record<RegionType, string> = {
  MainZone: "#2563eb",
  MarginTextZone: "#16a34a",
  NumberingZone: "#ca8a04",
  DamageZone: "#dc2626",
  GraphicZone: "#9333ea",
  DigitizationArtefactZone: "#6b7280",
  CustomZone: "#ec4899",
};
const REGION_HINTS: Record<RegionType, string> = {
  MainZone: "MainZone — primary text body (the manuscript's main column)",
  MarginTextZone: "MarginTextZone — marginal notes, glosses, or commentary around the main text",
  NumberingZone: "NumberingZone — folio/page/section numbers, quire signatures",
  DamageZone: "DamageZone — physically damaged or unreadable area (don't transcribe)",
  GraphicZone: "GraphicZone — illustration, decoration, or ornament",
  DigitizationArtefactZone: "DigitizationArtefactZone — scan artefacts: bleed-through, shadows, ruler marks",
  CustomZone: "CustomZone — any region not fitting the categories above",
};
const LINE_COLORS: Record<LineType, string> = {
  DefaultLine: "#111827",
  HeadingLine: "#2563eb",
  InterlinearLine: "#16a34a",
};
const LINE_HINTS: Record<LineType, string> = {
  DefaultLine: "DefaultLine — a normal line of text",
  HeadingLine: "HeadingLine — a heading, title, or rubric line",
  InterlinearLine: "InterlinearLine — a smaller line squeezed between two main lines",
};

const ACCENT = "#7c2d2d";

type Selection = { kind: "region" | "line"; id: string } | null;

function normalizePoint(p: Point | { x: number; y: number }): Point {
  if (Array.isArray(p)) return [Math.round(p[0]), Math.round(p[1])];
  return [Math.round(p.x), Math.round(p.y)];
}

function loadImageSize(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

function lineBounds(line: Line): { x: number; y: number; w: number; h: number } | null {
  if (line.baseline.length < 2) return null;
  const xs = line.baseline.map((p) => p[0]);
  const ys = line.baseline.map((p) => p[1]);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

export default function AnnotateEditor({ sessionId }: { sessionId: string }) {
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const viewerElRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const dragIndexRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [mode, setMode] = useState<Mode>("navigate");
  const [regionType, setRegionType] = useState<RegionType>("MainZone");
  const [lineType, setLineType] = useState<LineType>("DefaultLine");
  // ponytail: per-session baseline stroke width. Stored client-side only (not in
  // saved annotation state — that's a display concern, Kraken ignores it).
  const [lineWidth, setLineWidth] = useState<number>(3);
  const [regions, setRegions] = useState<Region[]>([]);
  const [lines, setLines] = useState<Line[]>([]);
  // ponytail: boot gate — block saves until the initial load has resolved.
  // Without this, a click before the fetch returns flips `dirty`, the 2s
  // autosave timer fires save() with the empty initial state, and the server
  // gets overwritten with [] — which is exactly the "always restarts empty" bug.
  const [booted, setBooted] = useState(false);
  // ponytail: per-vertex drag for the selected region — nudge nodes instead of redraw.
  // ceiling: fine for the common "small adjust" case; no edge-midpoint insert yet.
  const [dragVertex, setDragVertex] = useState<{ regionId: string; index: number } | null>(null);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [selected, setSelected] = useState<Selection>(null);
  const [status, setStatus] = useState("loading");
  const [dirty, setDirty] = useState(false);
  const [viewportTick, setViewportTick] = useState(0);
  const [twoClickActive, setTwoClickActive] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  // ponytail: per-session read direction (fix-a-v9-annotation-plan.md §9 #5).
  // Default horizontal-rl for Sogdian (kraken-fragmentary-manuscripts.md §4.2).
  // Client-side only — backend wiring is a separate task (B).
  const [readDirection, setReadDirection] = useState<"horizontal-rl" | "horizontal-lr">("horizontal-rl");
  // ponytail: image rotation degrees for the OSD viewer (§9 #4). OSD also has
  // its own showRotationControl buttons; this is an explicit app-level control.
  const [rotation, setRotation] = useState(0);
  // ponytail: keyboard-shortcut toggles (§9 #4). showMask = M, showOrder = L.
  // Neither renders heavy UI — showOrder paints a small index on each baseline;
  // showMask is a placeholder flag (no mask overlay implemented yet, YAGNI).
  const [showMask, setShowMask] = useState(false);
  const [showOrder, setShowOrder] = useState(false);
  // ponytail: Ctrl+A select-all lines (§9 #4). Our model is single-selection, so
  // this is a multi-select overlay: membership here highlights the row; clicking
  // any line clears it. No batch ops wired yet — minimal per the plan.
  const [selectedLineIds, setSelectedLineIds] = useState<Set<string>>(new Set());
  // ponytail: #10 per-point delete on baselines. Tracks the baseline vertex
  // being dragged (so Ctrl+Del knows which point to drop). Ceiling: only one
  // line's vertices are editable at a time (the selected line); no multi-line
  // vertex editing — upgrade to a per-line vertex layer if batch edits needed.
  const [activeVertexIndex, setActiveVertexIndex] = useState<number | null>(null);
  const [dragLineVertex, setDragLineVertex] = useState<{ lineId: string; index: number } | null>(null);
  // ponytail: #13 link/unlink cycle cursor — index into regions list for the
  // Y-key "cycle" when >1 region exists. Client-side only; reset on line change.
  const [linkCycle, setLinkCycle] = useState(0);
  // ponytail: #16 plain-text panel (Ctrl+5). Toggled side column showing all
  // transcripts as one editable block in reading order. Autosave on blur.
  const [showTextPanel, setShowTextPanel] = useState(false);

  const imageUrl = `/api/sessions/${sessionId}/image`;

  const selectedLine = useMemo(
    () => (selected?.kind === "line" ? lines.find((l) => l.id === selected.id) ?? null : null),
    [selected, lines],
  );
  const selectedLineIndex = useMemo(
    () => (selectedLine ? lines.findIndex((l) => l.id === selectedLine.id) : -1),
    [selectedLine, lines],
  );

  const markDirty = useCallback(() => {
    setDirty(true);
    setStatus("unsaved");
  }, []);

  const bumpViewport = useCallback(() => setViewportTick((v) => v + 1), []);

  // boot
  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const [size, savedRes] = await Promise.all([
          loadImageSize(imageUrl),
          fetch(`/api/sessions/${sessionId}/annotations`),
        ]);
        const saved: AnnotationState = savedRes.ok ? await savedRes.json() : { regions: [], lines: [] };
        let nextRegions = saved.regions ?? [];
        let nextLines = saved.lines ?? [];
        if (!nextRegions.length && !nextLines.length) {
          const sugRes = await fetch(`/api/sessions/${sessionId}/autosuggest`);
          if (sugRes.ok) {
            const sug = await sugRes.json();
            nextRegions = sug.regions ?? [];
            nextLines = sug.lines ?? [];
          }
        }
        if (disposed) return;

        setRegions(
          (nextRegions || [])
            .map((r, i) => ({
              id: r.id || `r${i + 1}`,
              polygon: (r.polygon || []).map(normalizePoint),
              type: (REGION_COLORS[r.type as RegionType] ? r.type : "MainZone") as RegionType,
            }))
            .filter((r) => r.polygon.length >= 3),
        );
        setLines(
          (nextLines || [])
            .map((l, i) => ({
              id: l.id || `l${i + 1}`,
              baseline: (l.baseline || []).map(normalizePoint),
              boundary: (l.boundary || []).map(normalizePoint),
              type: (LINE_COLORS[l.type as LineType] ? l.type : "DefaultLine") as LineType,
              transcript: l.transcript || "",
              // ponytail: preserve confidence if the server/proxy sends it (§9 #6).
              confidence: typeof (l as { confidence?: number | null }).confidence === "number"
                ? (l as { confidence?: number | null }).confidence
                : null,
              // ponytail: preserve explicit line→region link (§9 #13). Stale
              // refs (region since deleted) render as orphan; no cleanup here.
              regionId: typeof (l as { regionId?: string | null }).regionId === "string"
                ? (l as { regionId?: string | null }).regionId
                : null,
            }))
            .filter((l) => l.baseline.length >= 2),
        );
        // ponytail: release the save gate only after loaded state is committed.
        setBooted(true);

        const viewer = OpenSeadragon({
          id: "osd-viewer",
          prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.1/build/openseadragon/images/",
          showNavigator: true,
          showRotationControl: true,
          preserveViewport: true,
          gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false },
          // ponytail: OSD typings are incomplete for simple image sources — cast to any.
          tileSources: [
            {
              type: "image",
              url: imageUrl,
              buildPyramid: false,
              width: size.width,
              height: size.height,
            },
          ] as unknown as OpenSeadragon.Options["tileSources"],
        });
        viewer.addHandler("open", bumpViewport);
        viewer.addHandler("animation", bumpViewport);
        viewer.addHandler("resize", bumpViewport);
        viewer.addHandler("rotate", bumpViewport);
        viewerRef.current = viewer;
        setStatus("saved");
      } catch (e) {
        console.error(e);
        setStatus("load failed");
      }
    })();
    return () => {
      disposed = true;
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, [bumpViewport, imageUrl, sessionId]);

  const imageToScreen = useCallback(
    (point: Point): [number, number] | null => {
      const viewer = viewerRef.current;
      if (!viewer || viewer.world.getItemCount() === 0) return null;
      const tiled = viewer.world.getItemAt(0);
      const screen = tiled.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(point[0], point[1]),
      );
      return [screen.x, screen.y];
    },
    [viewportTick],
  );

  const screenToImage = useCallback((e: React.MouseEvent): Point | null => {
    const viewer = viewerRef.current;
    const el = viewerElRef.current;
    if (!viewer || viewer.world.getItemCount() === 0 || !el) return null;
    const rect = el.getBoundingClientRect();
    const vp = new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top);
    const ip = viewer.world.getItemAt(0).viewerElementToImageCoordinates(vp);
    return [Math.round(ip.x), Math.round(ip.y)];
  }, []);

  // ponytail: live vertex drag on the selected region. pointer events on the
  // whole svg would steal from pan-zoom, so we capture on the handle only.
  const onVertexPointerDown = useCallback(
    (e: React.PointerEvent, regionId: string, index: number) => {
      if (mode !== "region") return;
      e.stopPropagation();
      e.preventDefault();
      (e.target as Element).setPointerCapture?.(e.pointerId);
      setDragVertex({ regionId, index });
    },
    [mode],
  );
  const onVertexPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragVertex) return;
      e.preventDefault();
      const point = screenToImage(e as unknown as React.MouseEvent);
      if (!point) return;
      setRegions((items) =>
        items.map((r) => {
          if (r.id !== dragVertex.regionId) return r;
          const polygon = r.polygon.slice();
          polygon[dragVertex.index] = point;
          return { ...r, polygon };
        }),
      );
    },
    [dragVertex, screenToImage],
  );
  const onVertexPointerUp = useCallback((e: React.PointerEvent) => {
    if (!dragVertex) return;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    setDragVertex(null);
    markDirty();
  }, [dragVertex, markDirty]);

  // ponytail: #10 per-point delete + drag on baseline vertices. Mirror of the
  // region vertex drag, but on the selected line's baseline points. Reuses the
  // same pointer-capture pattern. Ceiling: only the selected line's vertices
  // are draggable; no insert-vertex (double-click to add) — YAGNI for small
  // fixes, upgrade to a polyline editor if curved baselines become common.
  const onLineVertexPointerDown = useCallback(
    (e: React.PointerEvent, lineId: string, index: number) => {
      e.stopPropagation();
      e.preventDefault();
      (e.target as Element).setPointerCapture?.(e.pointerId);
      setDragLineVertex({ lineId, index });
      setActiveVertexIndex(index);
    },
    [],
  );
  const onLineVertexPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragLineVertex) return;
      e.preventDefault();
      const point = screenToImage(e as unknown as React.MouseEvent);
      if (!point) return;
      setLines((items) =>
        items.map((l) => {
          if (l.id !== dragLineVertex.lineId) return l;
          const baseline = l.baseline.slice();
          baseline[dragLineVertex.index] = point;
          return { ...l, baseline };
        }),
      );
    },
    [dragLineVertex, screenToImage],
  );
  const onLineVertexPointerUp = useCallback((e: React.PointerEvent) => {
    if (!dragLineVertex) return;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    setDragLineVertex(null);
    markDirty();
  }, [dragLineVertex, markDirty]);

  // 2-click baseline (eScriptorium pattern)
  const handleOverlayClick = useCallback(
    (e: React.MouseEvent) => {
      if (mode === "navigate" || mode === "transcribe") return;
      const target = e.target as SVGElement;
      if (target.dataset.shape === "true") return;
      const point = screenToImage(e);
      if (!point) return;

      if (mode === "baseline") {
        // ponytail: 2-click create — fastest for the common straight-line case.
        if (!twoClickActive) {
          setDraftPoints([point]);
          setTwoClickActive(true);
        } else {
          const id = `l${Date.now()}`;
          const baseline = [draftPoints[0], point];
          setLines((items) => [...items, { id, baseline, boundary: [], type: lineType, transcript: "" }]);
          setSelected({ kind: "line", id });
          setDraftPoints([]);
          setTwoClickActive(false);
          markDirty();
        }
      } else if (mode === "region") {
        // region: click-to-add-points, double-click to finish (polygon needs N points)
        setDraftPoints((pts) => [...pts, point]);
      }
    },
    [mode, screenToImage, twoClickActive, draftPoints, lineType, markDirty],
  );

  const finishDraftRegion = useCallback(() => {
    if (mode === "region" && draftPoints.length >= 3) {
      const id = `r${Date.now()}`;
      setRegions((items) => [...items, { id, polygon: draftPoints, type: regionType }]);
      setSelected({ kind: "region", id });
      markDirty();
    }
    setDraftPoints([]);
  }, [mode, draftPoints, regionType, markDirty]);

  const save = useCallback(async () => {
    // ponytail: hard gate — never POST before the initial load resolves, or we
    // race the empty initial state onto the server and wipe saved work.
    if (!booted) return;
    setStatus("saving…");
    const res = await fetch(`/api/sessions/${sessionId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regions, lines }),
    });
    if (res.ok) {
      setDirty(false);
      setStatus(`saved ${new Date().toLocaleTimeString()}`);
    } else {
      setStatus("save failed");
    }
  }, [sessionId, regions, lines, booted]);

  const handleImportXml = useCallback(async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    if (!window.confirm("Importing XML will REPLACE the current regions and lines on this page. Continue?")) {
      // Reset the input so the same file can be re-selected later.
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    setStatus("importing…");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`/api/sessions/${sessionId}/import-xml`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const text = await res.text();
        console.error("import-xml failed:", text);
        setStatus("import failed");
        return;
      }
      const body = await res.json();
      // Reload annotations from server to get the canonical state.
      const reloadRes = await fetch(`/api/sessions/${sessionId}/annotations`);
      if (!reloadRes.ok) {
        setStatus("import failed — could not reload");
        return;
      }
      const saved: AnnotationState = await reloadRes.json();
      const nextRegions = saved.regions ?? [];
      const nextLines = saved.lines ?? [];
      setRegions(
        nextRegions
          .map((r, i) => ({
            id: r.id || `r${i + 1}`,
            polygon: (r.polygon || []).map(normalizePoint),
            type: (REGION_COLORS[r.type as RegionType] ? r.type : "MainZone") as RegionType,
          }))
          .filter((r) => r.polygon.length >= 3),
      );
      setLines(
        nextLines
          .map((l, i) => ({
            id: l.id || `l${i + 1}`,
            baseline: (l.baseline || []).map(normalizePoint),
            boundary: (l.boundary || []).map(normalizePoint),
            type: (LINE_COLORS[l.type as LineType] ? l.type : "DefaultLine") as LineType,
            transcript: l.transcript || "",
            // ponytail: preserve confidence on XML import too (§9 #6).
            confidence: typeof (l as { confidence?: number | null }).confidence === "number"
              ? (l as { confidence?: number | null }).confidence
              : null,
            // ponytail: preserve line→region link on XML import (§9 #13).
            regionId: typeof (l as { regionId?: string | null }).regionId === "string"
              ? (l as { regionId?: string | null }).regionId
              : null,
          }))
          .filter((l) => l.baseline.length >= 2),
      );
      setDirty(false);
      setStatus(`imported ${body.regions ?? "?"} regions, ${body.lines ?? "?"} lines`);
    } catch (err) {
      console.error("import-xml error:", err);
      setStatus("import failed");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [sessionId]);

  // autosave: 2s debounce (not 30s — too laggy)
  useEffect(() => {
    if (!booted || !dirty) return;
    const t = window.setTimeout(save, 2000);
    return () => window.clearTimeout(t);
  }, [booted, dirty, save]);

  // save on unload
  useEffect(() => {
    const handler = () => { if (booted && dirty) navigator.sendBeacon(`/api/sessions/${sessionId}/annotations`, JSON.stringify({ regions, lines })); };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty, regions, lines, sessionId]);

  // ponytail: delete is mode-scoped — Region mode touches regions only,
  // Baseline mode touches baselines only. Prevents accidentally nuking the
  // wrong kind when both a region and a baseline could match a click.
  function deleteSelected() {
    if (!selected) return;
    if (mode === "region" && selected.kind === "region") {
      setRegions((items) => items.filter((i) => i.id !== selected.id));
      setSelected(null);
      markDirty();
    } else if (mode === "baseline" && selected.kind === "line") {
      setLines((items) => items.filter((i) => i.id !== selected.id));
      setSelected(null);
      markDirty();
    }
  }

  function deleteAllBaselines() {
    if (lines.length === 0) return;
    if (!window.confirm(`Delete all ${lines.length} baselines? Regions are kept. This cannot be undone.`)) return;
    setLines([]);
    setSelected(null);
    markDirty();
  }

  // ponytail: ray-cast midpoint of baseline against region polygon.
  // MainZone regions from Kraken are simple rectangles, so midpoint test is sufficient;
  // upgrade to per-vertex test if curved/concave regions become common.
  function pointInPolygon(p: Point, polygon: Point[]): boolean {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
      const intersect = (yi > p[1]) !== (yj > p[1]) &&
        p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  // ponytail: #13 orphan = line with null/missing regionId, or a regionId that
  // no longer matches any region (stale ref after region deletion). Used for the
  // yellow dashed outline + ⚠ badge. Ceiling: O(lines×regions) per render —
  // fine for a single folio (~50 lines); upgrade to a Set lookup if it bites.
  function isOrphanLine(line: Line): boolean {
    if (!line.regionId) return true;
    return !regions.some((r) => r.id === line.regionId);
  }

  function baselineMidpoint(line: Line): Point {
    const pts = line.baseline;
    if (pts.length === 0) return [NaN, NaN];
    if (pts.length === 1) return pts[0];
    const mid = Math.floor(pts.length / 2);
    const a = pts[mid - 1], b = pts[mid];
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  }

  function deleteBaselinesInRegion() {
    if (!selected || selected.kind !== "region") return;
    const region = regions.find((r) => r.id === selected.id);
    if (!region) return;
    const before = lines.length;
    setLines((items) => items.filter((l) => !pointInPolygon(baselineMidpoint(l), region.polygon)));
    const removed = before - lines.length;
    if (removed > 0) markDirty();
  }

  function selectLine(id: string, focusTextarea: boolean) {
    setSelected({ kind: "line", id });
    setMode("transcribe");
    const line = lines.find((l) => l.id === id);
    if (line && viewerRef.current) {
      const b = lineBounds(line);
      if (b) {
        const viewer = viewerRef.current;
        const tiled = viewer.world.getItemAt(0);
        const rect = new OpenSeadragon.Rect(b.x, b.y, b.w, b.h);
        const vp = tiled.imageToViewportRectangle(rect);
        viewer.viewport.fitBoundsWithConstraints(vp, false);
      }
    }
    if (focusTextarea) {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function gotoLine(delta: number) {
    if (selectedLineIndex < 0) return;
    const next = selectedLineIndex + delta;
    if (next < 0 || next >= lines.length) return;
    selectLine(lines[next].id, true);
  }

  function updateTranscript(value: string) {
    if (!selectedLine) return;
    setLines((items) => items.map((l) => (l.id === selectedLine.id ? { ...l, transcript: value } : l)));
    markDirty();
  }

  function insertChar(char: string) {
    const ta = textareaRef.current;
    if (!ta || !selectedLine) return;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const next = ta.value.slice(0, start) + char + ta.value.slice(end);
    updateTranscript(next);
    requestAnimationFrame(() => {
      ta.focus();
      ta.selectionStart = start + char.length;
      ta.selectionEnd = start + char.length;
    });
  }

  function reorderLine(from: number, to: number) {
    if (from === to || from < 0 || to < 0 || to >= lines.length) return;
    setLines((items) => {
      const next = [...items];
      const [m] = next.splice(from, 1);
      next.splice(to, 0, m);
      return next;
    });
    markDirty();
  }

  // ponytail: rotate the OSD viewport (§9 #4). OSD's viewport.setRotation takes
  // absolute degrees. We keep `rotation` in state so the control stays in sync.
  function setViewerRotation(deg: number) {
    const v = viewerRef.current;
    if (v) v.viewport.setRotation(deg);
    setRotation(deg);
  }

  // ponytail: Ctrl+A select-all lines (§9 #4). Our model is single-selection,
  // so this populates a multi-select overlay set; clicking any line clears it.
  // No batch ops wired — minimal per the plan ("don't build new UI, just bind keys").
  function selectAllLines() {
    if (lines.length === 0) return;
    setSelectedLineIds(new Set(lines.map((l) => l.id)));
  }

  // ponytail: #10 Ctrl+Del — drop the active baseline vertex (the one last
  // dragged or hovered). Keeps the line if ≥2 points remain; below 2, the line
  // is invalid for Kraken so delete it outright (mirrors the load filter).
  function deleteActiveVertex() {
    if (activeVertexIndex == null || !selectedLine) return;
    const idx = activeVertexIndex;
    setLines((items) =>
      items
        .map((l) => {
          if (l.id !== selectedLine.id) return l;
          const baseline = l.baseline.slice(0, idx).concat(l.baseline.slice(idx + 1));
          return { ...l, baseline };
        })
        .filter((l) => l.baseline.length >= 2),
    );
    setActiveVertexIndex(null);
    markDirty();
  }

  // ponytail: #11 I-key invert — reverse baseline + boundary arrays of the
  // selected line. Marks dirty. Ceiling: only reverses point order; does not
  // recompute mask (Kraken recomputes on next segment) — fine for training.
  function invertSelectedLine() {
    if (!selectedLine) return;
    setLines((items) =>
      items.map((l) =>
        l.id === selectedLine.id
          ? { ...l, baseline: l.baseline.slice().reverse(), boundary: l.boundary.slice().reverse() }
          : l,
      ),
    );
    markDirty();
  }

  // ponytail: #12 J-key join — concat first two selected lines' baselines
  // (dedup shared endpoint), join transcripts with a space, drop the second.
  // Uses selectedLineIds if ≥2; else the selected line + the next line.
  // Ceiling: only joins two at a time; no polyline smoothing at the seam.
  function joinSelectedLines() {
    const ids =
      selectedLineIds.size >= 2
        ? Array.from(selectedLineIds).slice(0, 2)
        : selectedLine
          ? [selectedLine.id, lines[lines.findIndex((l) => l.id === selectedLine.id) + 1]?.id].filter(
              Boolean,
            ) as string[]
          : [];
    if (ids.length < 2) return;
    const a = lines.find((l) => l.id === ids[0]);
    const b = lines.find((l) => l.id === ids[1]);
    if (!a || !b) return;
    // dedup shared endpoint: if a's last point == b's first point, drop b's first.
    const aEnd = a.baseline[a.baseline.length - 1];
    const bStart = b.baseline[0];
    const sameEndpoint = aEnd && bStart && aEnd[0] === bStart[0] && aEnd[1] === bStart[1];
    const baseline = sameEndpoint
      ? a.baseline.concat(b.baseline.slice(1))
      : a.baseline.concat(b.baseline);
    const boundary = a.boundary.concat(b.boundary);
    const transcript = [a.transcript, b.transcript].filter((s) => s).join(" ");
    setLines((items) => {
      const next = items
        .map((l) => (l.id === a.id ? { ...l, baseline, boundary, transcript } : l))
        .filter((l) => l.id !== b.id);
      return next;
    });
    setSelectedLineIds(new Set());
    setSelected({ kind: "line", id: a.id });
    markDirty();
  }

  // ponytail: #13 Y-key link — assign selected line to a region. With one
  // region, link directly; with >1, cycle by linkCycle (reset on each line
  // change). Stale refs render as orphan; no server-side constraint.
  function linkSelectedLineToRegion() {
    if (!selectedLine || regions.length === 0) return;
    const target = regions.length === 1 ? regions[0] : regions[linkCycle % regions.length];
    setLinkCycle((c) => (regions.length > 1 ? c + 1 : c));
    setLines((items) => items.map((l) => (l.id === selectedLine.id ? { ...l, regionId: target.id } : l)));
    markDirty();
  }

  // ponytail: #13 U-key unlink — clear the selected line's regionId (→ orphan).
  function unlinkSelectedLine() {
    if (!selectedLine) return;
    setLines((items) => items.map((l) => (l.id === selectedLine.id ? { ...l, regionId: null } : l)));
    markDirty();
  }

  // ponytail: #14 Shift+L auto-sort — reorder lines by baseline centroid Y,
  // top-to-bottom. Reading order is computed (array index), never stored.
  // ponytail: not a menu item — one keystroke is the whole UX; a sort dialog
  // with column selection is YAGNI for a single-column Sogdian folio. Upgrade
  // path: add a column-picker if multi-column layouts appear.
  function autoSortByReadingOrder() {
    setLines((items) => {
      const withY = items.map((l) => {
        const ys = l.baseline.map((p) => p[1]);
        const cy = ys.length ? ys.reduce((s, y) => s + y, 0) / ys.length : 0;
        return { l, cy };
      });
      withY.sort((a, b) => a.cy - b.cy);
      return withY.map((x) => x.l);
    });
    markDirty();
  }

  // ponytail: M (mask toggle) — placeholder flag, no mask overlay rendered yet
  // (§9 #4). YAGNI until a real mask layer is needed; the binding is reserved.
  // ponytail: L (reading-order display toggle) — showOrder paints the index on
  // each baseline when on (rendered in the SVG below).

  // keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") {
        // Enter = save + next line (only in textarea)
        if (tag === "TEXTAREA" && e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          gotoLine(1);
        } else if (tag === "TEXTAREA" && e.key === "ArrowUp" && e.ctrlKey) {
          e.preventDefault();
          gotoLine(-1);
        } else if (tag === "TEXTAREA" && e.key === "ArrowDown" && e.ctrlKey) {
          e.preventDefault();
          gotoLine(1);
        }
        return;
      }
      if (e.key === "v") setMode("navigate");
      if (e.key === "r") setMode("region"); // ponytail: region mode now default-visible (§9 #2)
      if (e.key === "b") setMode("baseline");
      // ponytail: T is context-sensitive (§9 #4). In Region mode with a selected
      // region, T assigns the current regionType to it (eScriptorium "type assign").
      // Otherwise T switches to Transcribe (existing binding, preserved).
      if (e.key === "t") {
        if (mode === "region" && selected?.kind === "region") {
          setRegions((items) =>
            items.map((r) => (r.id === selected.id ? { ...r, type: regionType } : r)),
          );
          markDirty();
        } else {
          setMode("transcribe");
        }
      }
      // ponytail: M = mask toggle (placeholder), L = reading-order display (§9 #4).
      if (e.key === "m") setShowMask((v) => !v);
      if (e.key === "l") setShowOrder((v) => !v);
      if (e.key === "Escape") { setDraftPoints([]); setTwoClickActive(false); }
      if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
      if (e.ctrlKey && e.key.toLowerCase() === "s") { e.preventDefault(); save(); }
      // ponytail: Ctrl+A = select all lines (§9 #4). Prevent the browser default
      // (select page text) so the binding is reliable outside the textarea.
      if (e.ctrlKey && e.key.toLowerCase() === "a") { e.preventDefault(); selectAllLines(); }
      if (e.key === "ArrowDown" && selectedLineIndex >= 0) { e.preventDefault(); gotoLine(1); }
      if (e.key === "ArrowUp" && selectedLineIndex >= 0) { e.preventDefault(); gotoLine(-1); }
      // ponytail: Tier 1 shortcuts (§9 #10-16, #7). All additive — Tier 0
      // bindings above (v/r/b/t/m/l/Esc/Del/Backspace/Ctrl+S/Ctrl+A/arrows)
      // are untouched. Ctrl-prefixed additions land after the existing ones.
      // #10 per-point delete (Ctrl+Del) — only when a baseline vertex is active.
      if (e.ctrlKey && (e.key === "Delete" || e.key === "Backspace")) {
        e.preventDefault();
        deleteActiveVertex();
        return;
      }
      // #11 I — invert reading direction of selected line.
      if (e.key === "i" || e.key === "I") { invertSelectedLine(); }
      // #12 J — join first two selected lines (or selected + next).
      if (e.key === "j" || e.key === "J") { joinSelectedLines(); }
      // #13 Y link / U unlink selected line ↔ region.
      if (e.key === "y" || e.key === "Y") { linkSelectedLineToRegion(); }
      if (e.key === "u" || e.key === "U") { unlinkSelectedLine(); }
      // #14 Shift+L — auto-sort lines top-to-bottom by baseline centroid Y.
      // Plain L (reading-order display toggle) is the Tier 0 binding above.
      if (e.shiftKey && (e.key === "L" || e.key === "l")) {
        e.preventDefault();
        autoSortByReadingOrder();
      }
      // #15 ? — toggle help cheatsheet. Also Esc inside the modal closes it
      // (Esc's existing Tier 0 binding cancels drawing; both run, harmless).
      if (e.key === "?") { setShowHelp((v) => !v); }
      // #16 Ctrl+5 — toggle plain-text panel (all transcripts, reading order).
      if (e.ctrlKey && e.key === "5") { e.preventDefault(); setShowTextPanel((v) => !v); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const stats = useMemo(() => {
    const transcribed = lines.filter((l) => l.transcript.trim()).length;
    return `${transcribed}/${lines.length} lines${advanced ? ` · ${regions.length} regions` : ""}`;
  }, [lines, regions, advanced]);

  // ponytail: delete is mode-scoped, so the trash button is only live when
  // the current mode matches the selected shape's kind.
  const canDeleteSelected =
    !!selected &&
    ((mode === "region" && selected.kind === "region") ||
     (mode === "baseline" && selected.kind === "line"));

  function renderPoints(points: Point[]): string {
    return points
      .map(imageToScreen)
      .filter(Boolean)
      .map((p) => p!.join(","))
      .join(" ");
  }

  function renderShape(kind: "region" | "line", item: Region | Line) {
    const isSel = selected?.kind === kind && selected.id === item.id;
    const color = kind === "region" ? REGION_COLORS[(item as Region).type] : LINE_COLORS[(item as Line).type];
    const points = kind === "region" ? (item as Region).polygon : (item as Line).baseline;
    const common = {
      "data-shape": "true",
      className: `annotation-shape${isSel ? " selected" : ""}`,
      onClick: (e: React.MouseEvent) => {
        // ponytail: regions/baselines must not swallow clicks in other draw modes
        // (otherwise you can't start a baseline inside a region). Only handle
        // selection when in a mode that edits this shape's kind.
        if (kind === "region" && mode === "baseline") return;
        if (kind === "line" && mode === "region") return;
        e.stopPropagation();
        setSelected({ kind, id: item.id });
        if (kind === "line") {
          setMode("transcribe");
          const line = item as Line;
          const b = lineBounds(line);
          if (b && viewerRef.current) {
            const viewer = viewerRef.current;
            const tiled = viewer.world.getItemAt(0);
            const rect = new OpenSeadragon.Rect(b.x, b.y, b.w, b.h);
            const vp = tiled.imageToViewportRectangle(rect);
            viewer.viewport.fitBoundsWithConstraints(vp, false);
          }
        }
      },
    };
    if (kind === "region") {
      // ponytail: regions must not block baseline clicks inside them. In
      // baseline mode, drop pointer events so clicks fall through to the
      // overlay and start a baseline. (Selection is handled on the SVG parent
      // via the overlay's own click handler when not in baseline mode.)
      const regionStyle = mode === "baseline" ? { pointerEvents: "none" as const } : undefined;
      return (
        <polygon
          key={`r:${item.id}`}
          {...common}
          style={regionStyle}
          points={renderPoints(points)}
          fill={`${color}33`}
          stroke={color}
          strokeWidth={isSel ? 4 : 2}
        />
      );
    }
    return (
      <>
        <polyline
          key={`l:${item.id}`}
          {...common}
          points={renderPoints(points)}
          fill="none"
          stroke={color}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={isSel ? lineWidth + 2 : lineWidth}
        />
        {/* ponytail: #13 orphan indicator — yellow dashed outline over the
            baseline when the line has no live region link. A boundary outline
            would be better but most lines have empty boundary (Kraken fills it
            later), so we mark the baseline. Ceiling: visual only, no hit area. */}
        {kind === "line" && isOrphanLine(item as Line) && (
          <polyline
            key={`l:orph:${item.id}`}
            points={renderPoints(points)}
            fill="none"
            stroke="#eab308"
            strokeDasharray="5 3"
            strokeWidth={isSel ? lineWidth + 1 : lineWidth}
            style={{ pointerEvents: "none", opacity: 0.9 }}
          />
        )}
      </>
    );
  }

  const draftScreenPoints = draftPoints.map(imageToScreen).filter(Boolean) as [number, number][];

  const modes: { mode: Mode; label: string; icon: React.ReactNode; hint: string }[] = [
    { mode: "navigate", label: "Navigate", icon: <Cursor size={16} />, hint: "V — pan/zoom" },
    { mode: "region", label: "Region", icon: <PathIcon size={16} />, hint: "R — click points, double-click to close" },
    { mode: "baseline", label: "Baseline", icon: <ScribbleLoop size={16} />, hint: "B — click start, click end" },
    { mode: "transcribe", label: "Transcribe", icon: <TextAUnderline size={16} />, hint: "T — select a line to type" },
  ];
  // ponytail: Region mode (incl. DamageZone) is default-visible — fragmentary
  // folios need DamageZone for lacunae (kraken-fragmentary-manuscripts.md §4.1).
  // The `advanced` toggle now only gates the stats line and help section.
  const visibleModes = modes;

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <header className="h-12 flex items-center px-4 border-b border-stone-200 dark:border-stone-800 shrink-0 gap-4">
        <h1 className="text-sm font-medium text-stone-600 dark:text-stone-400 uppercase tracking-wider truncate">
          <a href="/" className="hover:text-accent">msocr</a> · <span className="font-mono text-accent">{sessionId}</span>
        </h1>
        <span className="text-xs text-stone-400">{stats}</span>
        <div className="flex-1" />
        <span className={`text-xs ${dirty ? "text-amber-600 dark:text-amber-400" : "text-stone-400"}`}>{status}</span>
        {/* ponytail: per-session read direction (§9 #5). Default horizontal-rl
            for Sogdian. Client-side only — backend wiring is a separate task (B). */}
        <select
          value={readDirection}
          onChange={(e) => setReadDirection(e.target.value as "horizontal-rl" | "horizontal-lr")}
          title="Per-session read direction (not yet saved to backend)"
          className="text-xs rounded-lg border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 px-2 py-1"
        >
          <option value="horizontal-rl">RTL</option>
          <option value="horizontal-lr">LTR</option>
        </select>
        <button
          onClick={() => setAdvanced((v) => !v)}
          title="Toggle region annotation tools (not needed for Kraken training)"
          className={`flex items-center justify-center w-7 h-7 text-xs rounded-lg border transition-colors ${advanced ? "bg-accent text-white border-accent" : "border-stone-300 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800"}`}
        >
          A
        </button>
        <button
          onClick={() => setShowHelp(true)}
          title="How to use this editor"
          className="flex items-center justify-center w-7 h-7 text-xs rounded-lg border border-stone-300 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
        >
          <Question size={14} />
        </button>
        <button
          onClick={save}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-stone-300 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
        >
          <FloppyDisk size={14} /> Save
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-stone-300 dark:border-stone-700 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
        >
          <FileArrowUp size={14} /> Import XML
        </button>
        <input
          type="file"
          ref={fileInputRef}
          accept=".xml,application/xml,text/xml"
          className="hidden"
          onChange={handleImportXml}
        />
        <a
          href={`/api/sessions/${sessionId}/export?format=page`}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-accent text-white hover:bg-accent/90 transition-colors"
        >
          <DownloadSimple size={14} /> PAGE XML
        </a>
      </header>

      <div className={`flex-1 grid grid-cols-1 overflow-hidden ${showTextPanel ? "lg:grid-cols-[56px_1fr_320px_360px]" : "lg:grid-cols-[56px_1fr_380px]"}`}>
        {/* toolbar rail */}
        <aside className="hidden lg:flex flex-col items-center py-3 gap-1 border-r border-stone-200 dark:border-stone-800 bg-stone-50 dark:bg-stone-950">
          {visibleModes.map((m) => (
            <button
              key={m.mode}
              onClick={() => { setMode(m.mode); setDraftPoints([]); setTwoClickActive(false); }}
              title={m.hint}
              className={`w-10 h-10 flex items-center justify-center rounded-lg transition-colors ${
                mode === m.mode
                  ? "bg-accent text-white"
                  : "text-stone-500 hover:bg-stone-200 dark:hover:bg-stone-800"
              }`}
            >
              {m.icon}
            </button>
          ))}
          <div className="flex-1" />
          {selected?.kind === "region" && (
            <button
              onClick={deleteBaselinesInRegion}
              title="Delete all baselines inside this region"
              className="flex items-center gap-1 px-2 h-10 rounded-lg text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors text-xs whitespace-nowrap"
            >
              <Eraser size={16} />
              <span className="hidden sm:inline">Clear baselines in region</span>
            </button>
          )}
          {mode === "baseline" && lines.length > 0 && (
            <button
              onClick={deleteAllBaselines}
              title="Delete every baseline on this page (regions kept)"
              className="flex items-center gap-1 px-2 h-10 rounded-lg text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors text-xs whitespace-nowrap"
            >
              <Trash size={16} />
              <span className="hidden sm:inline">Clear all baselines</span>
            </button>
          )}
          <button
            onClick={deleteSelected}
            disabled={!canDeleteSelected}
            title="Delete selected (Del)"
            className="w-10 h-10 flex items-center justify-center rounded-lg text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          >
            <Trash size={16} />
          </button>
        </aside>

        {/* mobile mode bar */}
        <div className="lg:hidden flex border-b border-stone-200 dark:border-stone-800">
          {visibleModes.map((m) => (
            <button
              key={m.mode}
              onClick={() => { setMode(m.mode); setDraftPoints([]); setTwoClickActive(false); }}
              className={`flex-1 py-2 text-xs flex items-center justify-center gap-1 ${
                mode === m.mode ? "bg-accent text-white" : "text-stone-500"
              }`}
            >
              {m.icon} {m.label}
            </button>
          ))}
        </div>

        {/* viewer */}
        <main ref={viewerElRef} className="relative bg-stone-200 dark:bg-stone-900 overflow-hidden">
          <div id="osd-viewer" className="absolute inset-0" />
          <svg
            className={`annotation-overlay absolute inset-0 w-full h-full ${mode}`}
            onClick={handleOverlayClick}
            onDoubleClick={(e) => { e.preventDefault(); if (mode === "region") finishDraftRegion(); }}
          >
            {regions.map((r) => renderShape("region", r))}
            {lines.map((l) => renderShape("line", l))}
            {/* ponytail: L key reading-order display (§9 #4) — paint the array
                index at each line's first baseline point when showOrder is on. */}
            {showOrder && lines.map((l, i) => {
              if (l.baseline.length === 0) return null;
              const sp = imageToScreen(l.baseline[0]);
              if (!sp) return null;
              return (
                <g key={`ord:${l.id}`}>
                  <circle cx={sp[0]} cy={sp[1]} r={11} fill={ACCENT} opacity={0.85} />
                  <text
                    x={sp[0]} y={sp[1]} dy={4}
                    textAnchor="middle"
                    fontSize={11}
                    fill="#fff"
                    fontWeight={700}
                    style={{ pointerEvents: "none", userSelect: "none" }}
                  >
                    {i + 1}
                  </text>
                </g>
              );
            })}
            {/* ponytail: vertex handles for the selected region in Region mode —
                drag to nudge nodes; no insert/delete-vertex yet (YAGNI for small fixes). */}
            {mode === "region" && selected?.kind === "region" && (() => {
              const r = regions.find((x) => x.id === selected.id);
              if (!r) return null;
              return r.polygon.map((p, i) => {
                const sp = imageToScreen(p);
                if (!sp) return null;
                return (
                  <circle
                    key={`v:${r.id}:${i}`}
                    data-shape="true"
                    className="vertex-handle"
                    cx={sp[0]}
                    cy={sp[1]}
                    r={6}
                    fill="#fff"
                    stroke={REGION_COLORS[r.type]}
                    strokeWidth={2}
                    style={{ cursor: dragVertex?.index === i ? "grabbing" : "grab" }}
                    onPointerDown={(e) => onVertexPointerDown(e, r.id, i)}
                    onPointerMove={onVertexPointerMove}
                    onPointerUp={onVertexPointerUp}
                    onPointerCancel={onVertexPointerUp}
                  />
                );
              });
            })()}
            {/* ponytail: #10 baseline vertex handles for the selected line — drag
                to nudge, Ctrl+Del on the last-touched vertex to delete it.
                Shown in Baseline and Transcribe modes so the user can fix points
                while transcribing. Ceiling: only the selected line; no insert. */}
            {selectedLine && (mode === "baseline" || mode === "transcribe") && (() => {
              const l = selectedLine;
              return l.baseline.map((p, i) => {
                const sp = imageToScreen(p);
                if (!sp) return null;
                const isActive = activeVertexIndex === i;
                return (
                  <circle
                    key={`lv:${l.id}:${i}`}
                    data-shape="true"
                    className="vertex-handle"
                    cx={sp[0]}
                    cy={sp[1]}
                    r={isActive ? 7 : 5}
                    fill={isActive ? ACCENT : "#fff"}
                    stroke={LINE_COLORS[l.type]}
                    strokeWidth={2}
                    style={{ cursor: dragLineVertex?.index === i ? "grabbing" : "grab" }}
                    onPointerDown={(e) => onLineVertexPointerDown(e, l.id, i)}
                    onPointerMove={onLineVertexPointerMove}
                    onPointerUp={onLineVertexPointerUp}
                    onPointerCancel={onLineVertexPointerUp}
                  />
                );
              });
            })()}
            {mode === "region" && draftScreenPoints.length > 1 && (
              <polyline
                className="draft-line"
                points={draftScreenPoints.map((p) => p.join(",")).join(" ")}
                fill="none"
                stroke={REGION_COLORS[regionType]}
                strokeDasharray="4 2"
                strokeWidth={2}
              />
            )}
            {mode === "baseline" && draftScreenPoints.length > 0 && twoClickActive && (
              <circle
                className="draft-point"
                cx={draftScreenPoints[0][0]}
                cy={draftScreenPoints[0][1]}
                r={5}
                fill={ACCENT}
              />
            )}
            {draftScreenPoints.map((p, i) => (
              <circle key={i} className="draft-point" cx={p[0]} cy={p[1]} r={4} fill={ACCENT} />
            ))}
          </svg>
          {/* type selector when active */}
          {mode === "region" && (
            <div className="absolute top-3 left-3 flex gap-1 bg-white/90 dark:bg-stone-900/90 backdrop-blur p-1 rounded-lg border border-stone-200 dark:border-stone-800">
              {REGION_TYPES.map((t) => (
                <button
                  key={t}
                  onClick={() => setRegionType(t)}
                  className={`hintable px-2 py-1 text-[10px] rounded ${
                    regionType === t ? "text-white" : "text-stone-500"
                  }`}
                  style={regionType === t ? { backgroundColor: REGION_COLORS[t] } : {}}
                  data-hint={REGION_HINTS[t]}
                >
                  {t.replace(/Zone$/, "")}
                </button>
              ))}
            </div>
          )}
          {mode === "baseline" && (
            <div className="absolute top-3 left-3 flex gap-1 items-center bg-white/90 dark:bg-stone-900/90 backdrop-blur p-1 rounded-lg border border-stone-200 dark:border-stone-800">
              {LINE_TYPES.map((t) => (
                <button
                  key={t}
                  onClick={() => setLineType(t)}
                  className={`hintable px-2 py-1 text-[10px] rounded ${
                    lineType === t ? "text-white" : "text-stone-500"
                  }`}
                  style={lineType === t ? { backgroundColor: LINE_COLORS[t] } : {}}
                  data-hint={LINE_HINTS[t]}
                >
                  {t.replace(/Line$/, "")}
                </button>
              ))}
              <label className="flex items-center gap-1 px-1 text-[10px] text-stone-500" title="Baseline stroke width">
                <span>w</span>
                <input
                  type="range"
                  min={1}
                  max={12}
                  step={1}
                  value={lineWidth}
                  onChange={(e) => setLineWidth(Number(e.target.value))}
                  className="w-16 accent-stone-700"
                />
                <span className="tabular-nums w-4 text-center">{lineWidth}</span>
              </label>
            </div>
          )}
          {/* ponytail: image rotation control (§9 #4). Calls OSD viewport.setRotation.
              Bottom-right so it doesn't fight the top-left type palettes. OSD also
              has its own rotation buttons (showRotationControl:true); this is explicit. */}
          <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-white/90 dark:bg-stone-900/90 backdrop-blur p-1 rounded-lg border border-stone-200 dark:border-stone-800">
            <button
              onClick={() => setViewerRotation(rotation - 90)}
              title="Rotate −90°"
              className="w-7 h-7 flex items-center justify-center rounded text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-800 text-xs"
            >
              ↺90
            </button>
            <span className="text-[10px] tabular-nums text-stone-500 w-10 text-center" title="Current rotation (degrees)">
              {rotation}°
            </span>
            <button
              onClick={() => setViewerRotation(rotation + 90)}
              title="Rotate +90°"
              className="w-7 h-7 flex items-center justify-center rounded text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-800 text-xs"
            >
              ↻90
            </button>
            <button
              onClick={() => setViewerRotation(0)}
              title="Reset rotation to 0°"
              className="w-7 h-7 flex items-center justify-center rounded text-stone-600 dark:text-stone-300 hover:bg-stone-200 dark:hover:bg-stone-800 text-xs"
            >
              0°
            </button>
          </div>
        </main>

        {/* transcription panel */}
        <aside className="flex flex-col border-l border-stone-200 dark:border-stone-800 bg-stone-50 dark:bg-stone-950 overflow-hidden">
          <div className="p-3 border-b border-stone-200 dark:border-stone-800 flex items-center gap-2">
            <span className="text-xs font-semibold uppercase text-stone-500">
              {selectedLine ? `Line ${selectedLineIndex + 1}` : "Transcription"}
            </span>
            {selectedLine && (
              <div className="flex gap-1 ml-auto">
                <button
                  onClick={() => gotoLine(-1)}
                  disabled={selectedLineIndex <= 0}
                  title="Previous (Ctrl+↑)"
                  className="p-1 rounded text-stone-500 hover:bg-stone-200 dark:hover:bg-stone-800 disabled:opacity-30"
                >
                  <ArrowUp size={14} />
                </button>
                <button
                  onClick={() => gotoLine(1)}
                  disabled={selectedLineIndex >= lines.length - 1}
                  title="Next (Enter / Ctrl+↓)"
                  className="p-1 rounded text-stone-500 hover:bg-stone-200 dark:hover:bg-stone-800 disabled:opacity-30"
                >
                  <ArrowDown size={14} />
                </button>
              </div>
            )}
          </div>

          <textarea
            ref={textareaRef}
            className="w-full p-3 text-sm leading-relaxed bg-white dark:bg-stone-900 outline-none resize-none border-b border-stone-200 dark:border-stone-800 min-h-[120px] font-mono"
            placeholder="Select a baseline, then enter Sogdian transcription (Sims-Williams Latin)…"
            value={selectedLine?.transcript ?? ""}
            disabled={!selectedLine}
            onChange={(e) => updateTranscript(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); gotoLine(1); }
              if (e.ctrlKey && e.key === "ArrowUp") { e.preventDefault(); gotoLine(-1); }
              if (e.ctrlKey && e.key === "ArrowDown") { e.preventDefault(); gotoLine(1); }
            }}
          />

          <div className="flex flex-wrap gap-1 p-2 border-b border-stone-200 dark:border-stone-800">
            {SOGDIAN_CHARS.map((c) => (
              <button
                key={c}
                type="button"
                disabled={!selectedLine}
                onClick={() => insertChar(c)}
                className="w-8 h-8 text-base rounded hover:bg-stone-200 dark:hover:bg-stone-800 disabled:opacity-30 font-mono"
              >
                {c}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto">
            <ol className="divide-y divide-stone-100 dark:divide-stone-800">
              {lines.map((line, i) => {
                // ponytail: Ctrl+A select-all overlay (§9 #4) — highlight rows in
                // the multi-select set; clicking any line clears it.
                const inSelAll = selectedLineIds.has(line.id);
                return (
                <li
                  key={line.id}
                  draggable
                  onDragStart={() => (dragIndexRef.current = i)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => { if (dragIndexRef.current !== null) reorderLine(dragIndexRef.current, i); dragIndexRef.current = null; }}
                  className={`p-2 cursor-pointer group ${
                    selectedLine?.id === line.id
                      ? "bg-accent/10"
                      : inSelAll
                        ? "bg-amber-50 dark:bg-amber-900/20"
                        : "hover:bg-stone-100 dark:hover:bg-stone-900"
                  }`}
                  onClick={() => { if (selectedLineIds.size > 0) setSelectedLineIds(new Set()); selectLine(line.id, true); }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-stone-400 w-5 shrink-0">{i + 1}</span>
                    {/* ponytail: #7 line type dropdown (§9 #7). Plan §9 lists
                        "Correction/Main/Numbering/Signature" but the existing
                        LineType enum is DefaultLine/HeadingLine/InterlinearLine
                        — the backend (PAGE XML emitter) already knows these
                        three, so we use them as-is rather than minting new
                        values the backend can't emit. Upgrade path: extend the
                        enum + backend together if the scholar needs more. */}
                    <select
                      value={line.type}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => {
                        e.stopPropagation();
                        const t = e.target.value as LineType;
                        setLines((items) => items.map((l) => (l.id === line.id ? { ...l, type: t } : l)));
                        markDirty();
                      }}
                      className="hintable text-[10px] rounded shrink-0 border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 px-1 py-0.5"
                      style={{ color: LINE_COLORS[line.type] }}
                      data-hint={LINE_HINTS[line.type]}
                      title={`Line type (§9 #7): ${LINE_HINTS[line.type]}`}
                    >
                      {LINE_TYPES.map((t) => (
                        <option key={t} value={t}>{t.replace(/Line$/, "")}</option>
                      ))}
                    </select>
                    {/* ponytail: #13 orphan ⚠ badge — line has no live region link. */}
                    {isOrphanLine(line) && (
                      <span
                        className="text-[10px] shrink-0"
                        title="Orphan line — not linked to any region (Y to link, U to unlink)"
                        style={{ color: "#eab308" }}
                      >
                        ⚠
                      </span>
                    )}
                    {/* ponytail: #13 region-link indicator — show the linked
                        region id (truncated) when present and live. */}
                    {line.regionId && regions.some((r) => r.id === line.regionId) && (
                      <span
                        className="text-[9px] font-mono px-1 rounded shrink-0"
                        title={`Linked to region ${line.regionId}`}
                        style={{ color: REGION_COLORS[regions.find((r) => r.id === line.regionId)!.type] }}
                      >
                        ◆
                      </span>
                    )}
                    {/* ponytail: per-line confidence placeholder (§9 #6). Shown
                        only when a numeric confidence is present — Tier 2 wires
                        real values. Color-coded: ≥0.9 green, ≥0.7 amber, else red. */}
                    {typeof line.confidence === "number" && (
                      <span
                        className="text-[9px] font-mono px-1 rounded shrink-0 tabular-nums"
                        title="Recognition confidence (0–1)"
                        style={{
                          color: line.confidence >= 0.9 ? "#16a34a"
                            : line.confidence >= 0.7 ? "#ca8a04" : "#dc2626",
                        }}
                      >
                        {(line.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className="text-[11px] text-stone-300 cursor-grab active:cursor-grabbing group-hover:text-stone-500 ml-auto" title="drag to reorder">⠿</span>
                  </div>
                  <div className="font-mono text-xs text-stone-700 dark:text-stone-300 mt-1 truncate text-left">
                    {line.transcript || <span className="italic text-stone-400">not transcribed</span>}
                  </div>
                </li>
                );
              })}
            </ol>
          </div>
        </aside>

        {/* ponytail: #16 plain-text panel (Ctrl+5). Toggled side column: all
            line transcripts as one editable block in reading order. Edit a row
            → update that line's transcript; autosave on blur (the global 2s
            debounce handles it). Hover a row → fit that line in the OSD viewer.
            ponytail: rendered as a togglable side column (not an overlay) so it
            doesn't cover the image — simplest layout that keeps both visible.
            Ceiling: one <textarea> per line re-renders all rows on each keystroke;
            fine for ~50 lines/folio, upgrade to a virtualized list if it stutters. */}
        {showTextPanel && (
          <aside className="hidden lg:flex flex-col border-l border-stone-200 dark:border-stone-800 bg-stone-50 dark:bg-stone-950 overflow-hidden">
            <div className="p-3 border-b border-stone-200 dark:border-stone-800 flex items-center gap-2">
              <span className="text-xs font-semibold uppercase text-stone-500">Plain text</span>
              <span className="text-[10px] text-stone-400">(Ctrl+5 to close)</span>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {lines.map((line, i) => (
                <div
                  key={line.id}
                  className="flex gap-1 items-start"
                  onMouseEnter={() => {
                    const b = lineBounds(line);
                    if (b && viewerRef.current) {
                      const viewer = viewerRef.current;
                      const tiled = viewer.world.getItemAt(0);
                      const rect = new OpenSeadragon.Rect(b.x, b.y, b.w, b.h);
                      const vp = tiled.imageToViewportRectangle(rect);
                      viewer.viewport.fitBoundsWithConstraints(vp, false);
                    }
                  }}
                >
                  <span className="text-[10px] font-mono text-stone-400 w-5 shrink-0 pt-1.5">{i + 1}</span>
                  <textarea
                    value={line.transcript}
                    onChange={(e) => {
                      const v = e.target.value;
                      setLines((items) => items.map((l) => (l.id === line.id ? { ...l, transcript: v } : l)));
                      markDirty();
                    }}
                    onBlur={() => save()}
                    className="flex-1 text-xs leading-relaxed bg-white dark:bg-stone-900 outline-none resize-none border border-stone-200 dark:border-stone-800 rounded p-1.5 min-h-[2rem] font-mono"
                    rows={1}
                    placeholder={`line ${i + 1}`}
                  />
                </div>
              ))}
              {lines.length === 0 && (
                <p className="text-xs text-stone-400 italic p-2">No lines yet.</p>
              )}
            </div>
          </aside>
        )}
      </div>

      {showHelp && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={() => setShowHelp(false)}
        >
          <div
            className="max-w-2xl max-h-[85vh] overflow-y-auto bg-white dark:bg-stone-900 rounded-2xl shadow-xl border border-stone-200 dark:border-stone-800 p-6 text-sm leading-relaxed"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">How to annotate</h2>
              <button
                onClick={() => setShowHelp(false)}
                className="p-1 rounded text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800"
              >
                <X size={18} />
              </button>
            </div>

            <p className="text-stone-600 dark:text-stone-400 mb-4">
              The page auto-segments on load (Kraken BLLA) — you review, fix, and transcribe.
              All changes auto-save after 2 seconds of inactivity.
            </p>

            <h3 className="font-semibold mt-4 mb-2">Workflow</h3>
            <ol className="list-decimal pl-5 space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>Regions</b> (R): optional — enclose text areas. Click points, double-click to close. Pick the right type from the top-left palette (DamageZone is now default-visible for fragmentary folios). <b>Not required for Kraken training.</b></li>
              <li><b>Baselines</b> (B): the reading line under each line of text. Click start, click end — that's it.</li>
              <li><b>Transcribe</b> (T): select a baseline, type the Sogdian text (Sims-Williams Latin transliteration) in the right panel. Press <code>Enter</code> to save and jump to the next line.</li>
              <li><b>Export</b>: click <i>PAGE XML</i> in the top bar to download for Kraken training.</li>
            </ol>

            <h3 className="font-semibold mt-4 mb-2">Modes</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>Navigate</b> (V) — pan and zoom the image. Hover any label for a 3-second tooltip.</li>
              <li><b>Region</b> (R) — draw region polygons. Top-left palette picks the type (incl. DamageZone).</li>
              <li><b>Baseline</b> (B) — 2-click to draw a baseline. Top-left palette picks the line type.</li>
              <li><b>Transcribe</b> (T) — type the Sims-Williams Latin transliteration in the right panel. Use the character palette below the textarea for aleph, ayin, dotted/special letters, and the Leiden underdot (U+0323) for uncertain glyphs.</li>
            </ul>

            {advanced && (<>
            <h3 className="font-semibold mt-4 mb-2">Region types</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>MainZone</b> — the primary text column.</li>
              <li><b>MarginTextZone</b> — marginal notes, glosses, commentary.</li>
              <li><b>NumberingZone</b> — folio / page / quire numbers.</li>
              <li><b>DamageZone</b> — physically damaged area; don't transcribe.</li>
              <li><b>GraphicZone</b> — illustrations, decorations.</li>
              <li><b>DigitizationArtefactZone</b> — scan bleed-through, shadows, ruler marks.</li>
              <li><b>CustomZone</b> — anything else.</li>
            </ul>
            </>)}

            <h3 className="font-semibold mt-4 mb-2">Line types</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>DefaultLine</b> — a normal line of text.</li>
              <li><b>HeadingLine</b> — a heading, title, or rubric.</li>
              <li><b>InterlinearLine</b> — a smaller line squeezed between two main lines.</li>
            </ul>

            <h3 className="font-semibold mt-4 mb-2">Keyboard</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><kbd>V</kbd> / <kbd>R</kbd> / <kbd>B</kbd> / <kbd>T</kbd> — switch modes (Navigate / Region / Baseline / Transcribe)</li>
              <li><kbd>T</kbd> in Region mode with a region selected — assign the current region type (eScriptorium "type assign")</li>
              <li><kbd>M</kbd> — toggle mask overlay (placeholder; no mask layer yet)</li>
              <li><kbd>L</kbd> — toggle reading-order display (number badges on baselines)</li>
              <li><kbd>Ctrl</kbd>+<kbd>A</kbd> — select all lines (highlight in the right panel; click any line to clear)</li>
              <li><kbd>Enter</kbd> (in textarea) — save + next line</li>
              <li><kbd>Ctrl</kbd>+<kbd>↑</kbd> / <kbd>↓</kbd> — previous / next line</li>
              <li><kbd>↑</kbd> / <kbd>↓</kbd> (no modifier) — previous / next line</li>
              <li><kbd>Esc</kbd> — cancel current drawing</li>
              <li><kbd>Del</kbd> / <kbd>Backspace</kbd> — delete selected (mode-scoped: Region mode → region, Baseline mode → baseline)</li>
              <li><kbd>Ctrl</kbd>+<kbd>S</kbd> — save now</li>
              <li><kbd>Ctrl</kbd>+<kbd>Del</kbd> — delete the active baseline vertex (drag a vertex first to mark it active; line kept if ≥2 points remain)</li>
              <li><kbd>I</kbd> — invert reading direction of the selected line (reverses baseline + boundary point order)</li>
              <li><kbd>J</kbd> — join the first two selected lines (concat baselines, dedup shared endpoint, join transcripts with a space, drop the second)</li>
              <li><kbd>Y</kbd> — link the selected line to a region (cycles through regions if more than one)</li>
              <li><kbd>U</kbd> — unlink the selected line from its region (→ orphan, yellow dashed outline + ⚠ badge)</li>
              <li><kbd>Shift</kbd>+<kbd>L</kbd> — auto-sort lines top-to-bottom by baseline centroid Y (reading order is the array index, never stored)</li>
              <li><kbd>?</kbd> — toggle this cheatsheet</li>
              <li><kbd>Ctrl</kbd>+<kbd>5</kbd> — toggle the plain-text panel (all transcripts in reading order; hover a row to fit that line in the viewer)</li>
            </ul>

            <h3 className="font-semibold mt-4 mb-2">Reordering lines</h3>
            <p className="text-stone-700 dark:text-stone-300">
              Drag the <span className="font-mono">⠿</span> handle on any line in the right panel to set reading order.
            </p>

            <button
              onClick={() => setShowHelp(false)}
              className="mt-6 px-4 py-2 rounded-lg bg-accent text-white text-xs w-full"
            >
              Got it
            </button>
          </div>
        </div>
      )}
    </div>
  );
}