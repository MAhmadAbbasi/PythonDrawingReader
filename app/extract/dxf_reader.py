"""DXF: rooms are closed polylines labelled by the room name inside them; electrical points are block
references (INSERT) mapped to point types through the drawing's legend and config/block_mapping.yaml."""
from __future__ import annotations

import io
import math
import re
import statistics
from collections import Counter

import ezdxf
from ezdxf import recover
from shapely.geometry import Point as SPoint, Polygon

from .common import (LEGEND_TITLE, PointItem, Room, Sheet, SymbolMapper, Text, floor_name, norm, parse_dimensions,
                     room_type, title_block)

# $INSUNITS -> sq ft per drawing unit²
UNIT_SQFT = {1: 1 / 144, 2: 1.0, 4: 1 / 92903.04, 5: 1 / 929.0304, 6: 10.7639}
UNIT_NAME = {1: "inches", 2: "feet", 4: "millimetres", 5: "centimetres", 6: "metres"}
ROOM_LAYER = re.compile(r"ROOM|AREA|SPACE|ZONE", re.I)


def _texts(msp) -> list[Text]:
    out = []
    for e in msp.query("TEXT MTEXT"):
        try:
            if e.dxftype() == "MTEXT":
                text, h = e.plain_text(), e.dxf.get("char_height", 1.0)
            else:
                text, h = e.dxf.text, e.dxf.get("height", 1.0)
            p = e.dxf.insert
            # TEXT aligned by its alignment point
            if e.dxftype() == "TEXT" and e.dxf.hasattr("align_point") and (e.dxf.get("halign", 0) or e.dxf.get("valign", 0)):
                p = e.dxf.align_point
            for i, line in enumerate(str(text).splitlines()):
                if line.strip():
                    out.append(Text(line.strip(), float(p[0]), float(p[1]) - i * h * 1.5, float(h or 1.0)))
        except Exception:
            continue
    return out


def _polygons(msp) -> list[tuple[Polygon, str]]:
    out = []
    for e in msp.query("LWPOLYLINE POLYLINE"):
        try:
            if e.dxftype() == "LWPOLYLINE":
                pts = [(float(x), float(y)) for x, y in e.get_points("xy")]
                closed = e.closed
            else:
                if e.is_3d_polyline or e.is_poly_face_mesh or e.is_polygon_mesh:
                    continue
                pts = [(float(v.dxf.location[0]), float(v.dxf.location[1])) for v in e.vertices]
                closed = e.is_closed
            if len(pts) >= 3 and (closed or math.dist(pts[0], pts[-1]) < 1e-6):
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area > 0 and poly.geom_type == "Polygon":
                    out.append((poly, e.dxf.layer))
        except Exception:
            continue
    return out


def _units(doc, polys: list[Polygon], log: list[str]) -> float:
    code = int(doc.header.get("$INSUNITS", 0) or 0)
    if code in UNIT_SQFT:
        log.append(f"Drawing units: {UNIT_NAME[code]}")
        return UNIT_SQFT[code]
    # not set: guess from the size of the rooms (a 12'x14' room = 168 sq ft)
    median = statistics.median([p.area for p in polys]) if polys else 0
    for limit, code in ((60, 6), (2000, 2), (60000, 1), (1.5e6, 5)):
        if median < limit:
            break
    else:
        code = 4
    log.append(f"Drawing units not set in the file - guessed {UNIT_NAME[code]} from the room sizes")
    return UNIT_SQFT[code]


def read(data: bytes, mapper: SymbolMapper, log: list[str], warnings: list[str]) -> tuple[dict, list[Sheet], list[dict]]:
    try:
        doc, auditor = recover.read(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - damaged file
        raise ValueError(f"The DXF file could not be read: {exc}")
    if auditor.has_errors:
        warnings.append(f"The DXF had {len(auditor.errors)} error(s) that were repaired while reading.")
    msp = doc.modelspace()
    log.append(f"DXF version {doc.dxfversion}, {len(msp)} entities in model space")

    texts = _texts(msp)
    # Step 1 - title block
    project = title_block(texts)
    log.append("Step 1 - project info: " + (", ".join(f"{k}={v}" for k, v in project.items()) or "no title block found"))

    # Step 2 - floors
    floor_titles = [(name, t.x, t.y) for t in texts if (name := floor_name(t.text))]
    log.append("Step 2 - floors: " + (", ".join(sorted({f[0] for f in floor_titles})) or "no floor titles - one floor"))

    # Step 3 - legend: blocks with a description to their right, under a LEGEND title
    inserts = list(msp.query("INSERT"))
    legend_map: dict[str, tuple[str, str]] = {}
    legend_handles: set[str] = set()
    legend_box = None
    for title in [t for t in texts if LEGEND_TITLE.match(norm(t.text))]:
        h = title.h or 1.0
        box = (title.x - 30 * h, title.y - 150 * h, title.x + 120 * h, title.y + h)
        for ins in inserts:
            x, y = float(ins.dxf.insert[0]), float(ins.dxf.insert[1])
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                continue
            desc = min((t for t in texts if 0 < t.x - x < 60 * h and abs(t.y - y) < 1.5 * max(t.h, h) and t is not title),
                       key=lambda t: t.x - x, default=None)
            if desc is None:
                continue
            kind = mapper.classify(desc.text, ins.dxf.name)
            legend_handles.add(ins.dxf.handle)
            if kind:
                legend_map[ins.dxf.name.upper()] = (desc.text, kind)
        legend_box = box
    legend = [{"symbol": name, "description": d, "type": k} for name, (d, k) in sorted(legend_map.items())]
    log.append(f"Step 3 - legend: {len(legend)} symbol(s)" + (": " + ", ".join(f"{l['symbol']}={l['type']}" for l in legend) if legend else
               " (no legend found - block names are matched with config/block_mapping.yaml)"))

    # Step 4 - rooms: the smallest closed polyline around each room name
    polys = _polygons(msp)
    sqft = _units(doc, [p for p, _ in polys], log)
    room_layer_polys = [(p, l) for p, l in polys if ROOM_LAYER.search(l or "")]
    candidates = room_layer_polys or polys
    candidates = [(p, l) for p, l in candidates if 15 <= p.area * sqft <= 20000]
    label_texts = [t for t in texts if not floor_name(t.text) and not LEGEND_TITLE.match(norm(t.text))
                   and not (legend_box and legend_box[0] <= t.x <= legend_box[2] and legend_box[1] <= t.y <= legend_box[3])]
    owners: dict[int, list[Text]] = {}
    for t in label_texts:
        sp = SPoint(t.x, t.y)
        inside = [(p.area, i) for i, (p, _) in enumerate(candidates) if p.covers(sp)]
        if inside:
            owners.setdefault(min(inside)[1], []).append(t)
    rooms: list[Room] = []
    for i, (poly, layer) in enumerate(candidates):
        own = owners.get(i, [])
        dims = next((d for t in own if (d := parse_dimensions(t.text))), None)
        named = [t for t in own if room_type(t.text)]
        other = [t for t in own if not parse_dimensions(t.text) and len(norm(t.text)) >= 3 and not re.fullmatch(r"[\d .]+", norm(t.text))]
        label = (named or other or [None])[0]
        if label is None and not room_layer_polys:
            continue   # an unlabelled outline (wall, furniture, plot boundary ...)
        c = poly.representative_point()
        rooms.append(Room(name=label.text.strip() if label else f"Room {len(rooms) + 1}", poly=poly, x=c.x, y=c.y,
                          kind=room_type(label.text) if label else None,
                          dimensions=dims[0] if dims else None,
                          area_sqft=dims[1] if dims else round(poly.area * sqft, 1)))
    log.append(f"Step 4 - rooms: {len(rooms)} found ({len(candidates)} closed outlines checked"
               + (", room layers used" if room_layer_polys else "") + ")")
    if not rooms:
        warnings.append("No rooms found: draw each room as a closed polyline with its name inside (a ROOM layer works best).")

    # Step 5 - points: every block reference outside the legend
    points: list[PointItem] = []
    unknown: Counter = Counter()
    for ins in inserts:
        if ins.dxf.handle in legend_handles:
            continue
        name = ins.dxf.name
        kind = legend_map.get(name.upper(), (None, None))[1] or mapper.classify(name, ins.dxf.layer,
                                                                              *[a.dxf.text for a in getattr(ins, "attribs", [])])
        if kind:
            points.append(PointItem(kind, float(ins.dxf.insert[0]), float(ins.dxf.insert[1])))
        elif kind is None and not name.startswith("*"):
            unknown[name] += 1
    counts = Counter(p.kind for p in points)
    log.append("Step 5 - points: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none found"))
    if unknown:
        top = ", ".join(f"{n} ({c})" for n, c in unknown.most_common(8))
        warnings.append(f"{sum(unknown.values())} block(s) are not mapped to a point type and were not counted: {top}. "
                        "Add them to config/block_mapping.yaml if they are electrical points.")
    if not points:
        warnings.append("No electrical symbols found. Symbols must be blocks (INSERT) - exploded symbols can't be counted.")

    sheet = Sheet(default_floor=(floor_titles[0][0] if len(floor_titles) == 1 else "Floor 1"),
                  rooms=rooms, points=points, floor_titles=floor_titles if len(floor_titles) > 1 else [])
    return project, [sheet], legend
