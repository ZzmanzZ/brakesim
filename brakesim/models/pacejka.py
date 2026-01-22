from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TirData:
    sections: dict[str, dict[str, object]]

    def get(self, section: str, key: str, default: float | None = None) -> float | None:
        value = self.sections.get(section, {}).get(key, default)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def _parse_value(raw: str) -> object:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    try:
        return float(text)
    except ValueError:
        return text


def parse_tir(path: str | Path) -> TirData:
    """
    Parse a .tir file into sections of key/value pairs.
    """
    current = None
    sections: dict[str, dict[str, object]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]")
            sections.setdefault(current, {})
            continue
        if "=" not in line or current is None:
            continue
        key, value = line.split("=", 1)
        sections[current][key.strip()] = _parse_value(value)
    return TirData(sections=sections)
