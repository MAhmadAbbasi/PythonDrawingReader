"""Which package building size (1 BHK ... 10 Kanal) a drawing is: from the size written on it, else from its
rooms (bedrooms for a small single-floor flat, floor area -> plot size for a house)."""
from __future__ import annotations

import math
import re

SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(MARLA|KANAL|BHK)\b", re.I)


def parse_size(name: str) -> tuple[str, float] | None:
    m = SIZE.search(name or "")
    return (m.group(2).upper(), float(m.group(1))) if m else None


def plot_sqft(kind: str, value: float, cfg: dict) -> float:
    marla = cfg.get("sqft_per_marla", 225)
    return value * marla * (cfg.get("marla_per_kanal", 20) if kind == "KANAL" else 1)


def _floor_stats(extraction: dict) -> tuple[int, float, int]:
    """floors with rooms (not the roof), the largest floor's room area, bedrooms"""
    floors, largest, bedrooms = 0, 0.0, 0
    for f in extraction.get("floors", []):
        rooms = [a for a in f.get("areas", []) if a.get("name") != "Unassigned"]
        if not rooms or f.get("name") == "Roof":
            continue
        floors += 1
        largest = max(largest, sum(a.get("areaSqft") or 0 for a in rooms))
        bedrooms += sum(1 for a in rooms if a.get("roomType") == "bedroom")
    return floors, largest, bedrooms


def choose(extraction: dict, sizes: list[dict], cfg: dict, override_id: int | None = None) -> dict | None:
    if not sizes:
        return None
    if override_id:
        s = next((s for s in sizes if s["id"] == override_id), None)
        if s:
            return {"id": s["id"], "name": s["name"], "reason": "chosen by you"}
    parsed = [(s, parse_size(s["name"])) for s in sizes]
    houses = [(s, p) for s, p in parsed if p and p[0] in ("MARLA", "KANAL")]
    flats = [(s, p) for s, p in parsed if p and p[0] == "BHK"]

    # 1) the size written on the drawing ("10 MARLA" in the title block)
    written = parse_size((extraction.get("project") or {}).get("plotSize") or "")
    if written:
        same = [(s, p) for s, p in parsed if p == written]
        if same:
            return {"id": same[0][0]["id"], "name": same[0][0]["name"], "reason": f"the drawing says {extraction['project']['plotSize']}"}
        if written[0] == "BHK" and flats:
            s, _ = min(flats, key=lambda sp: abs(sp[1][1] - written[1]))
            return {"id": s["id"], "name": s["name"], "reason": f"the drawing says {extraction['project']['plotSize']} - nearest size"}
        if houses:
            want = plot_sqft(*written, cfg)
            s, _ = min(houses, key=lambda sp: abs(math.log(plot_sqft(*sp[1], cfg) / want)))
            return {"id": s["id"], "name": s["name"], "reason": f"the drawing says {extraction['project']['plotSize']} - nearest size"}

    floors, largest, bedrooms = _floor_stats(extraction)
    # 2) a small single-floor plan with bedrooms: a flat
    if flats and bedrooms and (floors <= 1 and 0 < largest <= cfg.get("bhk_max_sqft", 1800) or not houses):
        s, _ = min(flats, key=lambda sp: abs(sp[1][1] - bedrooms))
        return {"id": s["id"], "name": s["name"], "reason": f"{bedrooms} bedroom(s) on a {round(largest)} sq ft floor"}
    # 3) a house: plot size from the largest floor's area
    if houses and largest > 0:
        plot = largest / cfg.get("floor_coverage", 0.8)
        s, p = min(houses, key=lambda sp: abs(math.log(plot_sqft(*sp[1], cfg) / plot)))
        return {"id": s["id"], "name": s["name"],
                "reason": f"largest floor {round(largest)} sq ft = about {round(plot)} sq ft plot ({round(plot / cfg.get('sqft_per_marla', 225), 1)} Marla)"}
    s = sizes[len(sizes) // 2]
    return {"id": s["id"], "name": s["name"], "reason": "the drawing has no room areas - choose the size yourself", "uncertain": True}
