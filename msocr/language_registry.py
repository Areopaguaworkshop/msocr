"""Canonical language metadata for the manuscript HTR runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    direction: str
    web_font: str
    iso: str
    aliases: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, str | tuple[str, ...]]:
        # Keep `font` as a compatibility mirror for older callers while standardizing
        # new code on `web_font`.
        return {
            "code": self.code,
            "direction": self.direction,
            "web_font": self.web_font,
            "font": self.web_font,
            "iso": self.iso,
            "aliases": self.aliases,
        }


_PROFILES = (
    LanguageProfile(
        "sogdian",
        "rtl",
        "Noto Sans Sogdian",
        "sog",
        aliases=("old_sogdian",),
    ),
)

LANGUAGE_REGISTRY: Dict[str, dict[str, str | tuple[str, ...]]] = {
    profile.code: profile.as_metadata() for profile in _PROFILES
}

_ALIASES = {
    alias: profile.code
    for profile in _PROFILES
    for alias in (profile.code, *profile.aliases)
}

CLI_LANGUAGE_CODES = ("sogdian",)

CLI_LANGUAGE_ALIASES = ("old_sogdian",)

DEMO_LANGUAGE_CODES = CLI_LANGUAGE_CODES


def normalize_language_code(value: str) -> str:
    key = value.strip().lower()
    if key not in _ALIASES:
        raise KeyError(key)
    return _ALIASES[key]


def is_supported_language(value: str) -> bool:
    return value.strip().lower() in _ALIASES


# Unicode block per script. Sogdian (U+10F30) and Syriac (U+0710) are the two
# script blocks this project trains HTR models for, per the design doc.
SCRIPT_BLOCKS = {
    "sogdian": "U+10F30",   # Sogdian block — Manichaean, Buddhist
    "syriac": "U+0710",     # Syriac block — Jingjiao/Christian Sogdian
}

VALID_SCRIPT_BLOCKS = set(SCRIPT_BLOCKS.values())

# Default Kraken fine-tune base model per script_block. Resolved relative to
# repo root at call time so missing models fail loudly with a real path.
# ponytail: a dict, not a config object. Add a block here when a new base
# model is downloaded; orchestrator picks it up automatically.
DEFAULT_BASE_MODELS: Dict[str, str] = {
    "U+0710": "models/kraken/sophro_mhiro_syriac.mlmodel",       # Syriac — Christian Sogdian (East Syriac/Nestorian)
    "U+10F30": "models/kraken/avestan_ms0040.mlmodel",           # Sogdian national script — closest analog (Middle Iranian, RTL)
}


def default_base_model_for_script_block(script_block: str) -> Optional[Path]:
    """Resolve the default fine-tune base model path for a script_block.

    Returns None if no default is registered. Caller decides whether to
    train from scratch or require an explicit --base-model.
    """
    rel = DEFAULT_BASE_MODELS.get(script_block)
    if rel is None:
        return None
    # ponytail: repo-root-relative; manifest.py uses the same REPO_ROOT convention.
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[1]
    return repo_root / rel


def script_block_for_language(lang: str) -> str:
    """Resolve the Unicode script block for a language code."""
    norm = normalize_language_code(lang)
    if norm not in SCRIPT_BLOCKS:
        raise ValueError(f"no script_block known for language: {lang}")
    return SCRIPT_BLOCKS[norm]
