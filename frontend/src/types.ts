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
}

export interface AnnotationState {
  regions: Region[];
  lines: Line[];
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