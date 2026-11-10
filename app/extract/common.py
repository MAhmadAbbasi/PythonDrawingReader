"""Shared pieces of the DXF and PDF readers: text rules, symbol mapping and putting points into rooms."""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely.geometry import Point as SPoint, Polygon

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def norm(text: str) -> str:
    """Upper case, symbols -> spaces, a space between letters and digits ("SKT15A" -> "SKT 15 A")."""
    s = re.sub(r"[^A-Za-z0-9]+", " ", text or "").upper()
    s = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── floors ──
FLOOR_PATTERNS = [
    (r"\bLOWER GROUND\b", "Lower Ground Floor"),
    (r"\bUPPER GROUND\b", "Upper Ground Floor"),
    (r"\bBASEMENT\b", "Basement"),
    (r"\bGROUND FLOOR\b|\bG ?F PLAN\b", "Ground Floor"),
    (r"\bMEZZ?ANINE\b|\bMEZ\b", "Mezzanine Floor"),
    (r"\bFIRST FLOOR\b|\b1 ?ST FLOOR\b", "First Floor"),
    (r"\bSECOND FLOOR\b|\b2 ?ND FLOOR\b", "Second Floor"),
    (r"\bTHIRD FLOOR\b|\b3 ?RD FLOOR\b", "Third Floor"),
    (r"\bROOF\b|\bTERRACE FLOOR\b|\bMUMTY\b", "Roof"),
]


def floor_name(text: str) -> str | None:
    n = norm(text)
    if len(n) > 60:
        return None
    for pattern, name in FLOOR_PATTERNS:
        # written together only for long words ("BASEMENTFLOOR"; not "ROOF" inside "WEATHERPROOF")
        if matches([pattern], n, min_letters=7):
            return name
    return None


def _joined(pattern: str, min_letters: int = 4) -> re.Pattern | None:
    """The pattern for text read with its spaces lost ("MASTERBEDROOM"); None for short abbreviations, which
    would match inside other words"""
    alts = []
    for alt in pattern.split("|"):                 # each alternative on its own ("ROOF|TERRACE FLOOR")
        q = alt.replace(r"\b", "").replace(" ?", "").replace(" ", "")
        if len(re.sub(r"[^A-Z]", "", q)) >= min_letters:
            alts.append(q)
    return re.compile("|".join(alts)) if alts else None


def matches(patterns: list[str], n: str, min_letters: int = 4) -> bool:
    """n (normalised text) matches one of the patterns - also when its words are written together"""
    if any(re.search(p, n) for p in patterns):
        return True
    joined = n.replace(" ", "")
    return any((j := _joined(p, min_letters)) is not None and j.search(joined) for p in patterns)


# ── rooms ──
ROOM_TYPES = [
    ("dressing", [r"\bDRESS"]),
    ("bath", [r"\bBATH", r"WASH ?ROOM", r"\bTOILET", r"\bW ?C\b", r"\bPOWDER", r"\bLAV", r"\bPWD\b"]),
    ("bedroom", [r"BED ?R(OO)?M", r"\bBED\b", r"\bMASTER\b", r"GUEST ROOM", r"\bKIDS?\b", r"\bB ?R\b"]),
    ("kitchen", [r"KITCHEN", r"PANTRY", r"\bKIT\b"]),
    ("dining", [r"DINING"]),
    ("lounge", [r"LOUNGE", r"LIVING", r"DRAWING", r"\bT ?V\b", r"FAMILY", r"SITTING", r"\bHALL\b"]),
    ("corridor", [r"CORRIDOR", r"PASSAGE", r"LOBBY", r"FOYER", r"ENTRANCE", r"GALLERY"]),
    ("stairs", [r"\bSTAIR", r"\bLIFT\b", r"\bELEVATOR\b", r"\bMUMTY\b"]),
    ("store", [r"\bSTORE", r"\bSTORAGE\b"]),
    ("garage", [r"GARAGE", r"\bPORCH\b", r"CAR PARK", r"PARKING"]),
    ("laundry", [r"LAUNDRY", r"\bWASH AREA\b", r"\bUTILITY\b"]),
    ("servant", [r"SERVANT", r"\bMAID\b", r"\bDRIVER\b", r"\bGUARD\b", r"\bSTAFF\b", r"\bQUARTERS?\b"]),
    ("study", [r"\bSTUDY\b", r"\bOFFICE\b", r"LIBRARY", r"\bPRAYER\b"]),
    ("lawn", [r"\bLAWN\b", r"\bGARDEN\b", r"\bCOURT ?YARD\b", r"\bPATIO\b"]),
    ("balcony", [r"BALCONY", r"\bTERRACE\b", r"\bVERANDAH?\b"]),
    ("other", [r"^ROOM \d+$"]),          # a space enclosed by walls whose name could not be read (pictures)
]


def room_type(text: str) -> str | None:
    n = norm(text)
    if not n or len(n) > 40:
        return None
    for kind, patterns in ROOM_TYPES:
        if any(re.search(p, n) for p in patterns):
            return kind
    for kind, patterns in ROOM_TYPES:        # words written together ("MASTERBEDROOM")
        if matches(patterns, n):
            return kind
    return None


# 12'x14'  12'-6" x 14'-0"  12 x 14  3.6m x 4.2m
_FT = r"(\d+(?:\.\d+)?)\s*(?:'|FT|FEET)?\s*(?:-?\s*(\d+(?:\.\d+)?)\s*(?:\"|''|IN))?"
DIM_FEET = re.compile(r"(?<![\d.])" + _FT + r"\s*[xX×*]\s*" + _FT + r"(?![\d.])")
DIM_METRE = re.compile(r"(\d+(?:\.\d+)?)\s*M\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*M\b", re.I)


def parse_dimensions(text: str) -> tuple[str, float] | None:
    """(the dimension text, area in sq ft) of a room size like 12'x14' - None if it isn't one"""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return None
    if re.search(r"SQ\W{0,2}MM|\bMM\b|MM2|\bCORE\b", t.upper()):
        return None                      # a cable size ("2x4 Sq.mm"), not a room
    m = DIM_METRE.search(t)
    if m:
        return t, round(float(m.group(1)) * float(m.group(2)) * 10.7639, 1)
    # fractions of an inch as decimals: 4½" / 4 1/2" -> 4.5"
    s = t
    for frac, dec in (("½", ".5"), ("¼", ".25"), ("¾", ".75")):
        s = re.sub(r"(\d)\s*" + frac, r"\g<1>" + dec, s)
    for frac, dec in (("1/2", ".5"), ("1/4", ".25"), ("3/4", ".75")):
        s = re.sub(r"(\d)[\s-]+" + frac + r"(?=\s*[\"″'])", r"\g<1>" + dec, s)
    # (OCR: an inch mark read as "*" - 11-7* X7-2")
    s = re.sub(r"(\d\s*-\s*\d+(?:\.\d+)?)\s*\*(?=\s*[xX×])", r'\1"', s)
    # feet-inches with the marks left out: 11-2 X 10-6" -> 11'-2" X 10'-6" (not 2 x 10'-6")
    s = re.sub(r"(?<![\d.])(\d{1,3})\s*['′]?\s*-\s*(\d{1,2}(?:\.\d+)?)(?![\d.\"″'])", "\\1'-\\2\"", s)
    m = DIM_FEET.search(s.upper())
    if not m:
        return None
    a = float(m.group(1)) + float(m.group(2) or 0) / 12
    b = float(m.group(3)) + float(m.group(4) or 0) / 12
    if not (2 <= a <= 200 and 2 <= b <= 200):
        return None
    return t, round(a * b, 1)


PLOT_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(MARLA|KANAL|BHK)\b", re.I)


def plot_size(text: str) -> str | None:
    m = PLOT_SIZE.search(text or "")
    if not m:
        return None
    unit = m.group(2).upper()
    number = m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1)
    return f"{number} {'BHK' if unit == 'BHK' else unit.title()}"


# ── title block ──
TITLE_FIELDS = {
    "projectName": [r"^PROJECT( NAME| TITLE)?\b", r"^TITLE\b", r"^NAME OF PROJECT\b"],
    "client": [r"^CLIENT\b", r"^OWNER\b", r"^CUSTOMER\b"],
    "drawingNumber": [r"^(DRAWING|DWG|SHEET) ?(NO|NUMBER|#)\b"],
    "scale": [r"^SCALE\b"],
    "date": [r"^DATE\b"],
    "architect": [r"^ARCHITECT\b", r"^ENGINEER\b", r"^DESIGNED BY\b", r"^CONSULTANT\b", r"^DRAWN BY\b"],
}


@dataclass
class Text:
    text: str
    x: float
    y: float
    h: float = 1.0


def title_block(texts: list[Text]) -> dict:
    """Project info: "PROJECT: Smith House" in one text, or the label with its value next to it"""
    info: dict[str, str] = {}
    for t in texts:
        n = norm(t.text)
        for key, patterns in TITLE_FIELDS.items():
            if key in info or not any(re.search(p, n) for p in patterns):
                continue
            raw = t.text.strip()
            value = raw.split(":", 1)[1].strip() if ":" in raw else ""
            if not value:   # "SCALE 1:100" - the rest of the text after the label
                m = next((m for pat in patterns if (m := re.match(pat.replace("^", "^\s*"), raw.upper()))), None)
                value = raw[m.end():].strip(" :-") if m else ""
            if not value:
                # the nearest text to the right on the same line, or just below
                best, dist = None, math.inf
                for o in texts:
                    if o is t or not o.text.strip():
                        continue
                    dx, dy = o.x - t.x, abs(o.y - t.y)
                    if 0 < dx < 60 * max(t.h, 1e-6) and dy < 0.8 * max(t.h, o.h):
                        d = dx
                    elif abs(dx) < 10 * t.h and 0 < dy < 2.5 * max(t.h, o.h):
                        d = 1000 * t.h + dy
                    else:
                        continue
                    if d < dist:
                        best, dist = o, d
                value = best.text.strip() if best else ""
            if value and not any(re.search(p, norm(value)) for ps in TITLE_FIELDS.values() for p in ps):
                info[key] = value[:150]
    for t in texts:
        size = plot_size(t.text)
        if size:
            info.setdefault("plotSize", size)
    return info


# ── symbols ──
class SymbolMapper:
    """Block name / legend description -> point type (light, fan, socket_5A ...), from block_mapping.yaml"""

    def __init__(self, path: Path | None = None):
        cfg = yaml.safe_load((path or CONFIG_DIR / "block_mapping.yaml").read_text(encoding="utf8")) or {}
        self.blocks = {str(k).upper(): v for k, v in (cfg.get("blocks") or {}).items()}
        self.types = [(name, [re.compile(p) for p in patterns]) for name, patterns in (cfg.get("types") or {}).items()]
        self.ignore = [re.compile(r"\b" + re.escape(w) + r"\b") for w in cfg.get("ignore") or []]
        self.types_list = [name for name, _ in self.types]
        self.joined = [(name, [j for p in patterns if (j := _joined(p)) is not None])
                       for name, patterns in (cfg.get("types") or {}).items()]

    def classify(self, *names: str) -> str | None:
        """The point type, "" when it is something to ignore (furniture, doors ...), None when unknown"""
        for name in names:
            if name and name.upper() in self.blocks:
                return self.blocks[name.upper()]
        for name in names:
            n = norm(name)
            if not n:
                continue
            for kind, patterns in self.types:
                if any(p.search(n) for p in patterns):
                    return kind
        for name in names:                    # words written together ("CEILINGFAN")
            n = norm(name).replace(" ", "")
            for kind, patterns in self.joined:
                if n and any(p.search(n) for p in patterns):
                    return kind
        for name in names:
            n = norm(name)
            if n and any(p.search(n) for p in self.ignore):
                return ""
        return None


LEGEND_TITLE = re.compile(r"^(ELECTRICAL )?(LEGENDS?|KEY|SYMBOLS?|SYMBOL LIST)\b")


# ── putting it together ──
@dataclass
class Room:
    name: str
    poly: Polygon | None      # None: only the label position is known (points go to the nearest label)
    x: float
    y: float
    kind: str | None = None
    dimensions: str | None = None
    area_sqft: float | None = None


@dataclass
class PointItem:
    kind: str
    x: float
    y: float


@dataclass
class Sheet:
    """One drawing sheet (DXF model space / one PDF page), in its own coordinates"""
    default_floor: str
    rooms: list[Room] = field(default_factory=list)
    points: list[PointItem] = field(default_factory=list)
    floor_titles: list[tuple[str, float, float]] = field(default_factory=list)
    confidence_penalty: float = 0.0        # e.g. PDF symbol matching is less certain than DXF blocks
    snap: float = 0.0                      # a point this close outside a room (drawn on its wall) is in that room
    debug: dict = field(default_factory=dict)   # what the reader saw, for the checking overlay
    # several drawings on one sheet: (floor / section name, (x0, y0, x1, y1)) of each floor plan or area plan
    views: list[tuple[str, tuple]] = field(default_factory=list)


def _nearest_floor(sheet: Sheet, x: float, y: float) -> str:
    if sheet.views:                       # the drawing it is drawn in
        inside = [(name, b) for name, b in sheet.views if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if inside:
            return min(inside, key=lambda v: (v[1][2] - v[1][0]) * (v[1][3] - v[1][1]))[0]
        return min(sheet.views, key=lambda v: (max(v[1][0] - x, 0, x - v[1][2]) ** 2 + max(v[1][1] - y, 0, y - v[1][3]) ** 2))[0]
    if not sheet.floor_titles:
        return sheet.default_floor
    return min(sheet.floor_titles, key=lambda f: (f[1] - x) ** 2 + (f[2] - y) ** 2)[0]


def _add(areas: list[dict], new: dict) -> None:
    """Adds a room read on one sheet. The same floor drawn on several sheets (lighting layout, power layout ...)
    gives the same room more than once: it is merged into the room of that name read on another sheet (the
    nearest one, when the name is used twice), keeping the highest count of each point type - a point drawn on
    two sheets is one point."""
    others = [a for a in areas if new["_sheet"] not in a.setdefault("_sheets", {a["_sheet"]})]
    same = [a for a in others if a["name"] == new["name"]]
    if not same and new["name"] != "Unassigned":
        # read differently on another sheet (OCR: "STAFF QUARTER," / "STAFF QUARTER") at the same place
        key = re.sub(r"[^A-Z]", "", new["name"].upper())
        same = [a for a in others if a["name"] != "Unassigned"
                and difflib.SequenceMatcher(None, key, re.sub(r"[^A-Z]", "", a["name"].upper())).ratio() >= 0.75
                and math.hypot(a["_pos"][0] - new["_pos"][0], a["_pos"][1] - new["_pos"][1]) < 60]
    if not same:
        areas.append(new)
        return
    old = min(same, key=lambda a: (a["_pos"][0] - new["_pos"][0]) ** 2 + (a["_pos"][1] - new["_pos"][1]) ** 2)
    old["_sheets"].add(new["_sheet"])
    for k, v in new["items"].items():
        old["items"][k] = max(old["items"].get(k, 0), v)
    old["items"] = dict(sorted(old["items"].items()))
    for key in ("areaSqft", "dimensions", "roomType"):
        if old[key] is None:
            old[key] = new[key]
    old["confidence"] = max(old["confidence"], new["confidence"])


def assemble(sheets: list[Sheet], log: list[str], warnings: list[str]) -> list[dict]:
    """Rooms and their point counts per floor, plus an "Unassigned" area for points outside every room"""
    floors: dict[str, list[dict]] = {}
    for sheet in sheets:                  # every floor plan drawn is listed, even when no room in it could be read
        for name, _ in sheet.views:
            floors.setdefault(name, [])
    for sheet_no, sheet in enumerate(sheets):
        areas: list[tuple[Room, dict, dict]] = []   # room, counts, info
        for room in sheet.rooms:
            areas.append((room, {}, {"floor": _nearest_floor(sheet, room.x, room.y)}))
        polys = [(i, a[0].poly) for i, a in enumerate(areas) if a[0].poly is not None]
        labels = [(i, a[0]) for i, a in enumerate(areas) if a[0].poly is None]
        # a room known only by its label takes the points near it (within a typical room's size)
        sizes = sorted(math.sqrt(p.area) for _, p in polys)
        reach = sizes[len(sizes) // 2] if sizes else math.inf
        unassigned: dict[str, dict[str, int]] = {}
        for p in sheet.points:
            sp = SPoint(p.x, p.y)
            inside = [(poly.area, i) for i, poly in polys if poly.covers(sp)]
            if not inside and sheet.snap:
                inside = [(poly.distance(sp), i) for i, poly in polys if poly.distance(sp) <= sheet.snap]
            if inside:
                idx = min(inside)[1]
            else:
                near = [(math.hypot(l.x - p.x, l.y - p.y), i) for i, l in labels]
                near = [n for n in near if n[0] <= reach]
                idx = min(near)[1] if near else None
            if idx is None:
                floor = _nearest_floor(sheet, p.x, p.y)
                unassigned.setdefault(floor, {})
                unassigned[floor][p.kind] = unassigned[floor].get(p.kind, 0) + 1
            else:
                counts = areas[idx][1]
                counts[p.kind] = counts.get(p.kind, 0) + 1
        for room, counts, info in areas:
            conf = 1.0 - sheet.confidence_penalty
            if room.poly is None:
                conf -= 0.3
            if room.area_sqft is None:
                conf -= 0.15
            if room.kind is None:
                conf -= 0.2
            if not counts:
                conf -= 0.1
            _add(floors.setdefault(info["floor"], []), {
                "name": room.name, "roomType": room.kind, "areaSqft": room.area_sqft, "dimensions": room.dimensions,
                "items": dict(sorted(counts.items())), "confidence": round(max(0.05, min(1.0, conf)), 2),
                "_sheet": sheet_no, "_pos": (room.x, room.y),
            })
        for floor, counts in unassigned.items():
            _add(floors.setdefault(floor, []), {
                "name": "Unassigned", "roomType": None, "areaSqft": None, "dimensions": None,
                "items": dict(sorted(counts.items())), "confidence": 0.0, "_sheet": sheet_no, "_pos": (0.0, 0.0),
            })
    for floor, areas in floors.items():
        for a in areas:
            a.pop("_sheet", None); a.pop("_pos", None); a.pop("_sheets", None)
        un = next((a for a in areas if a["name"] == "Unassigned"), None)
        if un:
            warnings.append(f"{floor}: {sum(un['items'].values())} point(s) are outside every room - they are in "
                            "\"Unassigned\"; move them to the right room.")
    order = [name for _, name in FLOOR_PATTERNS]
    result = []
    for name in sorted(floors, key=lambda f: (order.index(f) if f in order else len(order), f)):
        areas = floors[name]
        # same room name twice on a floor (two bedrooms labelled BED ROOM): number them
        seen: dict[str, int] = {}
        for a in areas:
            seen[a["name"]] = seen.get(a["name"], 0) + 1
        dup: dict[str, int] = {}
        for a in areas:
            if seen[a["name"]] > 1 and a["name"] != "Unassigned":
                dup[a["name"]] = dup.get(a["name"], 0) + 1
                a["name"] = f"{a['name']} {dup[a['name']]}"
        for a in areas:
            if a["name"] != "Unassigned" and a["confidence"] < 0.7:
                warnings.append(f"{name} / {a['name']}: low confidence ({a['confidence']}) - check its counts.")
        if not areas:
            warnings.append(f"{name}: drawn, but no room in it could be read - add its rooms by hand.")
        result.append({"name": name, "areas": areas,
                       "areaSqft": round(sum(a.get("areaSqft") or 0 for a in areas), 1)})
        log.append(f"{name}: {len([a for a in areas if a['name'] != 'Unassigned'])} room(s), "
                   f"{sum(sum(a['items'].values()) for a in areas)} point(s)")
    return result


def totals(floors: list[dict]) -> dict:
    items: dict[str, int] = {}
    sqft = 0.0
    for f in floors:
        for a in f["areas"]:
            sqft += a.get("areaSqft") or 0
            for k, v in a["items"].items():
                items[k] = items.get(k, 0) + v
    return {"areaSqft": round(sqft, 1), "items": dict(sorted(items.items())), "points": sum(items.values())}
