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

  const [mode, setMode] = useState<Mode>("navigate");
  const [regionType, setRegionType] = useState<RegionType>("MainZone");
  const [lineType, setLineType] = useState<LineType>("DefaultLine");
  const [regions, setRegions] = useState<Region[]>([]);
  const [lines, setLines] = useState<Line[]>([]);
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
            }))
            .filter((l) => l.baseline.length >= 2),
        );

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
  }, [sessionId, regions, lines]);

  // autosave: 2s debounce (not 30s — too laggy)
  useEffect(() => {
    if (!dirty) return;
    const t = window.setTimeout(save, 2000);
    return () => window.clearTimeout(t);
  }, [dirty, save]);

  // save on unload
  useEffect(() => {
    const handler = () => { if (dirty) navigator.sendBeacon(`/api/sessions/${sessionId}/annotations`, JSON.stringify({ regions, lines })); };
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
      if (e.key === "r") setMode("region");
      if (e.key === "b") setMode("baseline");
      if (e.key === "t") setMode("transcribe");
      if (e.key === "Escape") { setDraftPoints([]); setTwoClickActive(false); }
      if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
      if (e.ctrlKey && e.key.toLowerCase() === "s") { e.preventDefault(); save(); }
      if (e.key === "ArrowDown" && selectedLineIndex >= 0) { e.preventDefault(); gotoLine(1); }
      if (e.key === "ArrowUp" && selectedLineIndex >= 0) { e.preventDefault(); gotoLine(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const stats = useMemo(() => {
    const transcribed = lines.filter((l) => l.transcript.trim()).length;
    return `${transcribed}/${lines.length} lines · ${regions.length} regions`;
  }, [lines, regions]);

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
      return (
        <polygon
          key={`r:${item.id}`}
          {...common}
          points={renderPoints(points)}
          fill={`${color}33`}
          stroke={color}
          strokeWidth={isSel ? 4 : 2}
        />
      );
    }
    return (
      <polyline
        key={`l:${item.id}`}
        {...common}
        points={renderPoints(points)}
        fill="none"
        stroke={color}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={isSel ? 5 : 3}
      />
    );
  }

  const draftScreenPoints = draftPoints.map(imageToScreen).filter(Boolean) as [number, number][];

  const modes: { mode: Mode; label: string; icon: React.ReactNode; hint: string }[] = [
    { mode: "navigate", label: "Navigate", icon: <Cursor size={16} />, hint: "V — pan/zoom" },
    { mode: "region", label: "Region", icon: <PathIcon size={16} />, hint: "R — click points, double-click to close" },
    { mode: "baseline", label: "Baseline", icon: <ScribbleLoop size={16} />, hint: "B — click start, click end" },
    { mode: "transcribe", label: "Transcribe", icon: <TextAUnderline size={16} />, hint: "T — select a line to type" },
  ];

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <header className="h-12 flex items-center px-4 border-b border-stone-200 dark:border-stone-800 shrink-0 gap-4">
        <h1 className="text-sm font-medium text-stone-600 dark:text-stone-400 uppercase tracking-wider truncate">
          <a href="/" className="hover:text-accent">msocr</a> · <span className="font-mono text-accent">{sessionId}</span>
        </h1>
        <span className="text-xs text-stone-400">{stats}</span>
        <div className="flex-1" />
        <span className={`text-xs ${dirty ? "text-amber-600 dark:text-amber-400" : "text-stone-400"}`}>{status}</span>
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
        <a
          href={`/api/sessions/${sessionId}/export?format=page`}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-accent text-white hover:bg-accent/90 transition-colors"
        >
          <DownloadSimple size={14} /> PAGE XML
        </a>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[56px_1fr_380px] overflow-hidden">
        {/* toolbar rail */}
        <aside className="hidden lg:flex flex-col items-center py-3 gap-1 border-r border-stone-200 dark:border-stone-800 bg-stone-50 dark:bg-stone-950">
          {modes.map((m) => (
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
          {modes.map((m) => (
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
            <div className="absolute top-3 left-3 flex gap-1 bg-white/90 dark:bg-stone-900/90 backdrop-blur p-1 rounded-lg border border-stone-200 dark:border-stone-800">
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
            </div>
          )}
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
              {lines.map((line, i) => (
                <li
                  key={line.id}
                  draggable
                  onDragStart={() => (dragIndexRef.current = i)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => { if (dragIndexRef.current !== null) reorderLine(dragIndexRef.current, i); dragIndexRef.current = null; }}
                  className={`p-2 cursor-pointer group ${
                    selectedLine?.id === line.id ? "bg-accent/10" : "hover:bg-stone-100 dark:hover:bg-stone-900"
                  }`}
                  onClick={() => selectLine(line.id, true)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-stone-400 w-5 shrink-0">{i + 1}</span>
                    <span
                      className="hintable text-[10px] px-1 rounded shrink-0"
                      style={{ color: LINE_COLORS[line.type] }}
                      data-hint={LINE_HINTS[line.type]}
                    >
                      {line.type.replace(/Line$/, "")}
                    </span>
                    <span className="text-[11px] text-stone-300 cursor-grab active:cursor-grabbing group-hover:text-stone-500 ml-auto" title="drag to reorder">⠿</span>
                  </div>
                  <div className="font-mono text-xs text-stone-700 dark:text-stone-300 mt-1 truncate text-left">
                    {line.transcript || <span className="italic text-stone-400">not transcribed</span>}
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </aside>
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
              <li><b>Regions</b> (R): enclose text areas. Click points, double-click to close. Pick the right type from the top-left palette.</li>
              <li><b>Baselines</b> (B): the reading line under each line of text. Click start, click end — that's it.</li>
              <li><b>Transcribe</b> (T): select a baseline, type the Sogdian text (Sims-Williams Latin transliteration) in the right panel. Press <code>Enter</code> to save and jump to the next line.</li>
              <li><b>Export</b>: click <i>PAGE XML</i> in the top bar to download for Kraken training.</li>
            </ol>

            <h3 className="font-semibold mt-4 mb-2">Modes</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>Navigate</b> (V) — pan and zoom the image. Hover any label for a 3-second tooltip.</li>
              <li><b>Region</b> (R) — draw region polygons. Top-left palette picks the type.</li>
              <li><b>Baseline</b> (B) — 2-click to draw a baseline. Top-left palette picks the line type.</li>
              <li><b>Transcribe</b> (T) — type the Sims-Williams Latin transliteration in the right panel. Use the character palette below the textarea for aleph, ayin, dotted/special letters.</li>
            </ul>

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

            <h3 className="font-semibold mt-4 mb-2">Line types</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><b>DefaultLine</b> — a normal line of text.</li>
              <li><b>HeadingLine</b> — a heading, title, or rubric.</li>
              <li><b>InterlinearLine</b> — a smaller line squeezed between two main lines.</li>
            </ul>

            <h3 className="font-semibold mt-4 mb-2">Keyboard</h3>
            <ul className="space-y-1 text-stone-700 dark:text-stone-300">
              <li><kbd>V</kbd> / <kbd>R</kbd> / <kbd>B</kbd> / <kbd>T</kbd> — switch modes</li>
              <li><kbd>Enter</kbd> (in textarea) — save + next line</li>
              <li><kbd>Ctrl</kbd>+<kbd>↑</kbd> / <kbd>↓</kbd> — previous / next line</li>
              <li><kbd>↑</kbd> / <kbd>↓</kbd> (no modifier) — previous / next line</li>
              <li><kbd>Esc</kbd> — cancel current drawing</li>
              <li><kbd>Del</kbd> / <kbd>Backspace</kbd> — delete selected (mode-scoped: Region mode → region, Baseline mode → baseline)</li>
              <li><kbd>Ctrl</kbd>+<kbd>S</kbd> — save now</li>
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