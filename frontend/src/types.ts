export type Point = [number, number];

export type RegionType =
  | "MainZone"
  | "MarginTextZone"
  | "NumberingZone"
  | "DamageZone"
  | "GraphicZone"
  | "DigitizationArtefactZone"
  | "CustomZone";

export type LineType = "DefaultLine" | "HeadingLine" | "InterlinearLine";

export type Mode = "navigate" | "region" | "baseline" | "transcribe";

export interface Region {
  id: string;
  polygon: Point[];
  type: RegionType;
}

export interface Line {
  id: string;
  baseline: Point[];
  boundary: Point[];
  type: LineType;
  transcript: string;
  // ponytail: per-line recognition confidence placeholder (0–1, null = not yet
  // recognized). Tier 2 task will wire real values from /autosuggest; UI must
  // not break if this field appears. (fix-a-v9-annotation-plan.md §9 #6.)
  confidence?: number | null;
  // ponytail: explicit line→region link (§9 #13). Additive optional field.
  // null/missing or stale ref = orphan line (rendered with yellow dashed
  // outline + ⚠ badge). Existing session JSON without this field still loads.
  regionId?: string | null;
  // A logical row may contain several physically disconnected, independently
  // trainable line fragments. Index 0 is first in RTL reading order.
  rowId?: string | null;
  fragmentIndex?: number | null;
  trainable?: boolean;
  exclusionReason?: string | null;
}

export interface Gap {
  id: string;
  rowId: string;
  afterLineId: string;
  beforeLineId: string;
  type: "hole" | "tear" | "lost";
  polygon: Point[];
  confidence: number | null;
}

export interface AnnotationState {
  regions: Region[];
  lines: Line[];
  gaps: Gap[];
}

export interface SessionSummary {
  session_id: string;
  language: string;
  script_variant: string;
  source: string;
  line_count: number;
  updated_at: string;
}

export interface LanguageInfo {
  code: string;
  direction: string;
  web_font: string;
}

export interface PlateInfo {
  plate_number: number;
  filename: string;
  path: string;
  thumbnail_url: string;
  status: {
    session_id: string;
    line_count: number;
    transcribed_count: number;
    updated_at: string;
  } | null;
}
