"""Canonical language metadata shared across CLI, services, and docs-facing APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
    LanguageProfile("syriac", "rtl", "Noto Sans Syriac", "syr"),
    LanguageProfile("sogdian", "rtl", "Noto Sans Sogdian", "sog"),
    LanguageProfile("old_sogdian", "rtl", "Noto Sans Old Sogdian", "sog"),
    LanguageProfile("old_turkish", "rtl", "Noto Sans Old Turkic", "otk"),
    LanguageProfile("greek", "ltr", "GFS Didot, Noto Serif", "grc"),
    LanguageProfile("latin", "ltr", "Junicode, EB Garamond", "lat"),
    LanguageProfile("coptic", "ltr", "Noto Sans Coptic, Antinoou", "cop"),
    LanguageProfile("armenian", "ltr", "Noto Sans Armenian", "hye", aliases=("armenia",)),
    LanguageProfile("geez", "ltr", "Noto Sans Ethiopic", "gez"),
)

LANGUAGE_REGISTRY: Dict[str, dict[str, str | tuple[str, ...]]] = {
    profile.code: profile.as_metadata() for profile in _PROFILES
}

_ALIASES = {
    alias: profile.code
    for profile in _PROFILES
    for alias in (profile.code, *profile.aliases)
}

CLI_LANGUAGE_CODES = (
    "greek",
    "latin",
    "syriac",
    "coptic",
    "armenian",
    "geez",
    "sogdian",
    "old_turkish",
)

CLI_LANGUAGE_ALIASES = ("armenia",)

DEMO_LANGUAGE_CODES = (
    "greek",
    "latin",
    "syriac",
    "coptic",
    "armenian",
    "geez",
)


def normalize_language_code(value: str) -> str:
    key = value.strip().lower()
    if key not in _ALIASES:
        raise KeyError(key)
    return _ALIASES[key]


def is_supported_language(value: str) -> bool:
    return value.strip().lower() in _ALIASES
