"""Vector PDF exported from CAD: each page is a floor (or several floors with their titles).

Per page:
  1. text lines (room names, dimensions, legend, title block) and the drawing broken into strokes
  2. legend: each description row and the symbol drawn to its left (a shape, a shape with a letter, or a short
     text like "AC")
  3. symbols: small strokes that touch are grouped into symbols and matched to the closest legend symbol by
     shape, at any rotation (see pdf_geometry)
  4. rooms: the enclosed space around each room name, traced from the wall lines with door openings closed;
     else the smallest drawn rectangle around the name; else only the name's position
  5. areas: room dimensions written on the plan ("14'x12'"), else outline x drawing scale (from "SCALE 1:100",
     or worked out from the rooms that have dimensions written)"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait

import cv2
import numpy as np
import pymupdf
import shapely
from shapely.geometry import box

from . import ocr
from . import pdf_geometry as geo
from .common import (LEGEND_TITLE, TITLE_FIELDS, PointItem, Room, Sheet, SymbolMapper, Text, floor_name, norm,
                     parse_dimensions, room_type, title_block)

SCALE = re.compile(r"1\s*:\s*(\d+)")
# 1/8" = 1'-0"   (inches on paper per foot)
SCALE_IMPERIAL = re.compile(r"(\d+)\s*/\s*(\d+)\s*(?:\"|''|”|″)?\s*=\s*1\s*(?:'|’|′)")


def _scale(texts: list[Text]) -> int | None:
    """the drawing scale as 1:N, from "SCALE 1:100" or "SCALE: 1/8"=1'-0"" """
    for t in texts:
        if "SCALE" not in norm(t.text):
            continue
        if m := SCALE_IMPERIAL.search(t.text):
            num, den = int(m.group(1)), int(m.group(2))
            if num and den:
                return round(12 * den / num)
        if m := SCALE.search(t.text):
            return int(m.group(1))
    return None


def _lines(page) -> list[tuple[Text, pymupdf.Rect]]:
    """text lines in the coordinates of the page as shown (PyMuPDF gives them unrotated)"""
    m = page.rotation_matrix
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            # spans far apart on one line are separate texts (two room names side by side)
            groups: list[list[dict]] = []
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                if groups and span["bbox"][0] - groups[-1][-1]["bbox"][2] < 1.5 * span["size"]:
                    groups[-1].append(span)
                else:
                    groups.append([span])
            for spans in groups:
                text = " ".join(s["text"].strip() for s in spans)
                r = (pymupdf.Rect(spans[0]["bbox"]) | pymupdf.Rect(spans[-1]["bbox"])) * m
                out.append((Text(text, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, max(s["size"] for s in spans)), r))
    return out


def _is_word(text: str) -> bool:
    """a word that can be part of a room name on another line (2 letters only if a room word itself: "TV")"""
    n = norm(text)
    if not (len(n) <= 20 and re.fullmatch(r"[A-Z ]+", n)):
        return False
    # a room word itself, or a word that goes with one ("GUEST" / "ROOM"); not a nearby note ("PVC CONDUIT")
    return bool(room_type(n)) or bool(set(n.split()) & NAME_WORDS)


NAME_WORDS = {"ROOM", "AREA", "HALL", "QUARTER", "QUARTERS", "AND", "GUEST", "STAFF", "DRESS", "FAMILY", "ATTACHED",
              "MAIN", "UPPER", "LOWER", "CHILDREN", "SON", "DAUGHTER", "PARENTS", "GRAND", "WALK", "IN", "CLOSET"}
NOTE_WORDS = re.compile(r"^(?:PVC|CONDUIT|CONDUI|CABLE|LPDB\w*|SQMM|WIRE|FROM|TO)+")


def _clean_name(name: str) -> str:
    """circuit tags read together with the name ("1S/EY1STAFF QUARTER" -> "STAFF QUARTER")"""
    words = []
    for w in name.split():
        stripped = NOTE_WORDS.sub("", w)      # a cable note read together with the name ("PVCCONDUITPASSAGE")
        if stripped != w and room_type(stripped):
            w = stripped
        w = w.strip(",.;:()[]{}")
        if not w:
            continue
        if w.isdigit():                       # "BED ROOM 2"
            words.append(w)
            continue
        if re.search(r"[\d/]", w):
            tail = re.sub(r"^.*[^A-Z]", "", w.upper())      # the letters after the tag: "1S/EY1STAFF" -> "STAFF"
            if len(tail) >= 3 and tail.isalpha():
                words.append(tail)
            continue
        words.append(w)
    return " ".join(words) or name


def _title_text(text: str) -> bool:
    n = norm(text)
    return any(re.search(p, n) for ps in TITLE_FIELDS.values() for p in ps)


def _room_labels(lines, legend_rect, skip: set[int]) -> list[tuple[str, str, pymupdf.Rect, float]]:
    """(name, room type, text box, text height); a name on two lines ("MASTER" / "BED ROOM") is one label.
    skip: texts that are symbols ("AC", the "TV" inside a TV point symbol)"""
    dims = [r for t, r in lines if parse_dimensions(t.text)]

    def sized(r) -> bool:                 # a room size written right under it ("MUMTY" / "11'-7" X 7'-2"")
        # (-0.5: OCR boxes can overlap a little)
        return any(-0.5 * r.height <= d.y0 - r.y1 < 2.5 * r.height and abs((d.x0 + d.x1) / 2 - (r.x0 + r.x1) / 2) < max(d.width, r.width)
                   for d in dims)

    free = [(t, r) for k, (t, r) in enumerate(lines) if k not in skip and not (legend_rect and legend_rect.intersects(r))
            and (not floor_name(t.text) or (room_type(t.text) and sized(r))) and not _title_text(t.text)]
    used: set[int] = set()
    labels = []
    for i, (t, r) in enumerate(free):
        kind = room_type(t.text)
        if not kind or i in used:
            continue
        parts, rect = [(r.y0, t.text.strip())], pymupdf.Rect(r)
        used.add(i)
        grew = True
        while grew:                    # words stacked right above / below the name
            grew = False
            for j, (o, orr) in enumerate(free):
                if j in used or not _is_word(o.text) or parse_dimensions(o.text):
                    continue
                h = max(rect.height / max(len(parts), 1), orr.height)
                stacked = min(abs(orr.y0 - rect.y1), abs(rect.y0 - orr.y1)) < 0.6 * h
                centred = abs((orr.x0 + orr.x1) / 2 - (rect.x0 + rect.x1) / 2) < max(rect.width, orr.width) / 2 + h
                if stacked and centred:
                    parts.append((orr.y0, o.text.strip()))
                    rect |= orr
                    used.add(j)
                    grew = True
        name = _clean_name(" ".join(text for _, text in sorted(parts)))
        labels.append((name, room_type(name) or kind, rect, t.h))
    return labels


def _legend(lines, items: list[geo.Stroke], mapper: SymbolMapper):
    """legend symbols as shapes, as short texts, the legend's area and its rows for the log"""
    shapes: list[tuple[geo.Symbol, str]] = []
    texts: dict[str, str] = {}
    rows_out: dict[str, dict] = {}
    legend_rect = None
    for title, tr in [(t, r) for t, r in lines if LEGEND_TITLE.match(norm(t.text))]:
        # the rows under the title: texts starting near the title's left edge, one after another
        below = sorted(((t, r) for t, r in lines if r.y0 > tr.y0 + 1 and tr.x0 - 4 * tr.height < r.x0 < tr.x0 + 6 * tr.height),
                       key=lambda tr_: tr_[1].y0)
        rows, last = [], tr
        for t, r in below:
            if r.y0 - last.y1 > 3 * r.height:
                break
            rows.append((t, r)); last = r
        if not rows:
            continue
        area = pymupdf.Rect(tr)
        for t, r in rows:
            area |= r
        for t, r in rows:
            kind = mapper.classify(t.text)
            if not kind:
                continue
            hh = r.height
            strip = (r.x0 - 6 * hh, r.y0 - 1.5 * hh, r.x0 + 1, r.y1 + 1.5 * hh)
            candidates = [s for s in geo.find_symbols([s for s in items if s.inside(strip)], 4 * hh, 0.15 * hh)
                          if s.bbox[2] <= r.x0 + 1 and abs(s.y - (r.y0 + r.y1) / 2) < hh]
            short = [(o, orr) for o, orr in lines if orr.x1 <= r.x0 + 1 and r.x0 - orr.x1 < 6 * hh
                     and abs(orr.y0 - r.y0) < hh / 2 and len(o.text.strip()) <= 4]
            if candidates:
                sym = max(candidates, key=lambda s: s.bbox[2])
                sym.text = _inner_text(sym, short)
                shapes.append((sym, kind))
                area |= pymupdf.Rect(sym.bbox)
                rows_out.setdefault(t.text, {"symbol": "shape" + (f" with \"{sym.text}\"" if sym.text else ""),
                                             "description": t.text, "type": kind})
            elif short:
                texts[norm(short[0][0].text)] = kind
                area |= short[0][1]
                rows_out.setdefault(t.text, {"symbol": short[0][0].text, "description": t.text, "type": kind})
        legend_rect = (legend_rect | area) if legend_rect else area
    table = _legend_table(lines, items, mapper)
    if table:                                  # a table legend: the row reading above saw its cells wrongly
        shapes, texts, rows_out, legend_rect = [], {}, {}, None
    for sym, kind, desc, area in table:
        shapes.append((sym, kind))
        rows_out.setdefault(desc, {"symbol": "shape", "description": desc, "type": kind})
        legend_rect = (legend_rect | area) if legend_rect else pymupdf.Rect(area)
    if legend_rect:
        legend_rect = legend_rect + (-2, -2, 2, 2)
    if len(shapes) >= 4:              # bits much smaller than the other legend symbols (a letter, a tick) match junk
        typical = statistics.median(sym.size for sym, _ in shapes)
        shapes = [(sym, kind) for sym, kind in shapes if sym.size >= 0.45 * typical]
    return shapes, texts, legend_rect, rows_out


def _legend_table(lines, items: list[geo.Stroke], mapper: SymbolMapper):
    """A legend drawn as a table: a SYMBOL(S) column and a DESCRIPTION column. Every symbol drawn in the symbol
    column of a row counts as that row's type (a row can show several: normal / smart / dimmer switch)."""
    out = []
    heads = [(t, r) for t, r in lines if re.fullmatch(r"SYMBOLS?", norm(t.text))]
    descs = [(t, r) for t, r in lines if norm(t.text).startswith("DESCRIPTION")]
    for sh, sr in heads:
        cy = (sr.y0 + sr.y1) / 2
        right = [(t, r) for t, r in descs if r.x0 > sr.x1 and r.x0 - sr.x1 < 60 * sr.height and abs((r.y0 + r.y1) / 2 - cy) < sr.height]
        if not right:
            continue
        dt, dr = min(right, key=lambda tr: tr[1].x0)
        # description texts: below the headers, starting between the two header columns
        cells = sorted(((t, r) for t, r in lines if r.y0 > sr.y1 and sr.x1 < r.x0 < (dr.x0 + dr.x1) / 2),
                       key=lambda tr: tr[1].y0)
        rows: list[list[tuple[Text, pymupdf.Rect]]] = []
        last_y = sr.y1
        for t, r in cells:
            if rows and abs((r.y0 + r.y1) / 2 - (rows[-1][0][1].y0 + rows[-1][0][1].y1) / 2) < r.height / 2:
                rows[-1].append((t, r))           # more words on the same row
                continue
            if r.y0 - last_y > 8 * max(r.height, 2):
                break                             # end of the table
            rows.append([(t, r)])
            last_y = r.y1
        if not rows:
            continue
        centres = [(row[0][1].y0 + row[0][1].y1) / 2 for row in rows]
        tx0, tx1 = sr.x0 - 2 * sr.width, max(r.x0 for row in rows for _, r in row)
        ty0, ty1 = sr.y1 - 1, max(r.y1 for _, r in rows[-1]) + 12
        table_items = [s for s in items if s.bbox[0] >= tx0 and s.bbox[2] <= tx1 and s.bbox[1] >= ty0 and s.bbox[3] <= ty1]
        pitch = statistics.median(np.diff(centres)) if len(centres) > 1 else 3 * rows[0][0][1].height
        left = sr.x0 - 1.5 * sr.width
        for row, y in zip(rows, centres):
            desc = " ".join(t.text for t, _ in sorted(row, key=lambda tr: tr[1].x0))
            kind = mapper.classify(desc)
            if not kind:
                continue
            x_end = min(r.x0 for _, r in row) - 0.5
            strip = (left, y - 0.5 * pitch, x_end, y + 0.5 * pitch)
            # symbols can be wider than the row is high (a distribution board); table borders are long lines
            found = geo.find_symbols([s for s in table_items if s.inside(strip)], 0.9 * (x_end - left), 0.04 * pitch,
                                     max_line=0.8 * pitch)
            if not found:
                continue
            biggest = max(s.size for s in found)
            for sym in found:
                if sym.size >= 0.4 * biggest:
                    out.append((sym, kind, desc, pymupdf.Rect(sr) | dr | pymupdf.Rect(strip) | row[-1][1]))
    return out


# titles of the drawings on a sheet that are not floor plans (floor plans are named by floor_name)
VIEW_TITLE = [
    (re.compile(r"\bELEVATIONS?\b"), "elevation"),
    (re.compile(r"\bSECTIONS?\b(?! CUT)"), "section"),
    (re.compile(r"\bAREA PLAN\b|\bAREA CHART\b"), "area"),
    (re.compile(r"\bSITE PLAN\b|\bLOCATION PLAN\b|\bKEY PLAN\b"), "site"),
    (re.compile(r"\bDETAILS?\b"), "detail"),
]
NOT_PLAN = {"elevation", "section", "site", "detail"}


def _view_title(text: str) -> tuple[str, str] | None:
    """(name, kind) when the text is the title of a drawing on the sheet: a floor plan ("GROUND FLOOR PLAN",
    kind "plan"), an elevation, a section, an area plan ..."""
    n = norm(text)
    if not n or len(n.split()) > 7:
        return None
    for pattern, kind in VIEW_TITLE:
        if pattern.search(n):
            return " ".join(w.capitalize() for w in n.split()), kind
    if floor := floor_name(text):
        return floor, "plan"
    return None


def _views(items: list[geo.Stroke], lines, page_rect, legend_rect) -> list[dict]:
    """The separate drawings on a sheet (several floor plans, an elevation, a section, an area plan side by side):
    blocks of drawing with clear paper between them, each named by the title written under (or in) it.
    [{name, kind: plan | area | elevation | section | site | detail | other, bbox, titled}]"""
    W, H = page_rect.width, page_rect.height
    g = max(1.0, min(W, H) / 600)                  # grid cell (points)
    gw, gh = int(W / g) + 2, int(H / g) + 2
    canvas = np.zeros((gh, gw), np.uint8)
    for s in items:
        x0, y0, x1, y1 = s.bbox
        if x1 - x0 > 0.8 * W or y1 - y0 > 0.8 * H:
            continue                               # the sheet frame and title block borders
        if legend_rect and s.inside(tuple(legend_rect)):
            continue
        pts = np.round(np.asarray(s.pts) / g).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 1, 1)
    gap = max(3, int(0.012 * min(W, H) / g))
    joined = cv2.dilate(canvas, np.ones((gap, gap), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if min(w, h) * g >= 0.12 * min(W, H):     # (a title block strip or a note is not a drawing)
            boxes.append([x * g, y * g, (x + w) * g, (y + h) * g])
    # drawings overlapping each other are one
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    titles = [(vt, r) for t, r in lines if (vt := _view_title(t.text)) and not (legend_rect and legend_rect.intersects(r))]
    # two drawings drawn close together join into one block: two titles side by side under it split it again
    split = []
    for b in boxes:
        under = sorted((r.x0 + r.x1) / 2 for _, r in titles
                       if b[0] <= (r.x0 + r.x1) / 2 <= b[2] and -0.25 * (b[3] - b[1]) <= (r.y0 + r.y1) / 2 - b[3] <= 0.12 * H)
        xs = [x for i, x in enumerate(under) if i == 0 or x - under[i - 1] > 0.25 * (b[2] - b[0])]
        cuts = [b[0]] + [(a + c) / 2 for a, c in zip(xs, xs[1:])] + [b[2]]
        split += [[cuts[i], b[1], cuts[i + 1], b[3]] for i in range(len(cuts) - 1)]
    views = [{"bbox": tuple(b), "name": None, "kind": "other", "titled": False} for b in split]
    # titles: under the drawing (or inside it, near its foot), across its middle; the nearest such drawing
    for (name, kind), r in sorted(titles, key=lambda tr: -tr[1].height):
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        best = None
        for v in views:
            x0, y0, x1, y1 = v["bbox"]
            if v["titled"] or not (x0 - 0.1 * (x1 - x0) <= cx <= x1 + 0.1 * (x1 - x0)):
                continue
            below = cy - y1                         # under its foot (negative: inside it)
            if -0.25 * (y1 - y0) <= below <= 0.12 * H:
                d = abs(below)
                if best is None or d < best[0]:
                    best = (d, v)
        if best is None:                           # a title written beside the drawing: the nearest one, close by
            for v in views:
                x0, y0, x1, y1 = v["bbox"]
                d = ((max(x0 - cx, 0, cx - x1)) ** 2 + (max(y0 - cy, 0, cy - y1)) ** 2) ** 0.5
                if not v["titled"] and d <= 0.15 * W and (best is None or d < best[0]):
                    best = (d, v)
        if best:
            best[1].update(name=name, kind=kind, titled=True)
    return views


def legend_rows(lines, mapper: SymbolMapper) -> list[tuple[str, str, pymupdf.Rect]]:
    """(type, description, where its symbol is drawn) for each legend row - for pictures, whose symbols are
    matched as images (raster_reader). A table legend (SYMBOL | DESCRIPTION) wins over plain rows."""
    table: list[tuple[str, str, pymupdf.Rect]] = []
    heads = [(t, r) for t, r in lines if re.fullmatch(r"SYMBOLS?", norm(t.text))]
    descs = [(t, r) for t, r in lines if norm(t.text).startswith("DESCRIPTION")]
    for sh, sr in heads:
        cy = (sr.y0 + sr.y1) / 2
        right = [(t, r) for t, r in descs if r.x0 > sr.x1 and r.x0 - sr.x1 < 60 * sr.height and abs((r.y0 + r.y1) / 2 - cy) < sr.height]
        if not right:
            continue
        dt, dr = min(right, key=lambda tr: tr[1].x0)
        cells = sorted(((t, r) for t, r in lines if r.y0 > sr.y1 and sr.x1 < r.x0 < (dr.x0 + dr.x1) / 2), key=lambda tr: tr[1].y0)
        rows: list[list] = []
        last_y = sr.y1
        for t, r in cells:
            if rows and abs((r.y0 + r.y1) / 2 - (rows[-1][0][1].y0 + rows[-1][0][1].y1) / 2) < r.height / 2:
                rows[-1].append((t, r))
                continue
            if r.y0 - last_y > 8 * max(r.height, 2):
                break
            rows.append([(t, r)])
            last_y = r.y1
        if not rows:
            continue
        centres = [(row[0][1].y0 + row[0][1].y1) / 2 for row in rows]
        pitch = statistics.median(np.diff(centres)) if len(centres) > 1 else 3 * rows[0][0][1].height
        left = sr.x0 - 1.5 * sr.width
        for row, y in zip(rows, centres):
            desc = " ".join(t.text for t, _ in sorted(row, key=lambda tr: tr[1].x0))
            kind = mapper.classify(desc)
            if kind:
                x_end = min(r.x0 for _, r in row) - 0.5
                table.append((kind, desc, pymupdf.Rect(left, y - 0.5 * pitch, x_end, y + 0.5 * pitch)))
    if table:
        return table
    out = []
    for title, tr in [(t, r) for t, r in lines if LEGEND_TITLE.match(norm(t.text))]:
        below = sorted(((t, r) for t, r in lines if r.y0 > tr.y0 + 1 and tr.x0 - 4 * tr.height < r.x0 < tr.x0 + 6 * tr.height),
                       key=lambda tr_: tr_[1].y0)
        rows, last = [], tr
        for t, r in below:
            if r.y0 - last.y1 > 3 * r.height:
                break
            last = r
            # a short text with a description to its right on the same line is a symbol ("AC"), not a row
            if len(t.text.strip()) <= 4 and any(o is not t and o2.x0 > r.x1 and o2.x0 - r.x1 < 8 * r.height
                                                and abs((o2.y0 + o2.y1) / 2 - (r.y0 + r.y1) / 2) < r.height / 2
                                                for o, o2 in lines):
                continue
            rows.append((t, r))
        # each row's symbol lies left of its text, up to halfway to the rows above and below
        mids = [(r.y0 + r.y1) / 2 for _, r in rows]
        for i, (t, r) in enumerate(rows):
            kind = mapper.classify(t.text)
            if not kind:
                continue
            hh = r.height
            top = (mids[i - 1] + mids[i]) / 2 if i else mids[i] - 1.5 * hh
            bottom = (mids[i] + mids[i + 1]) / 2 if i + 1 < len(rows) else mids[i] + 1.5 * hh
            out.append((kind, t.text, pymupdf.Rect(r.x0 - 6 * hh, max(top, mids[i] - 1.5 * hh), r.x0 - 0.5,
                                                   min(bottom, mids[i] + 1.5 * hh))))
    return out


def _dots(sym: geo.Symbol, dot_kinds: dict[str, tuple[str, float]]) -> tuple[str, int] | None:
    """(type, how many) when the symbol is a blob of touching dots of a dot-drawn legend type"""
    if not sym.filled or sym.color not in dot_kinds:
        return None
    kind, one = dot_kinds[sym.color]
    n = round(sym.fill_area / one)
    return (kind, n) if 1 <= n <= 12 and abs(sym.fill_area / one - n) < 0.35 else None


def _inner_text(sym: geo.Symbol, lines) -> str:
    x0, y0, x1, y1 = sym.bbox
    inner = [t for t, _ in lines if len(t.text.strip()) <= 4 and x0 <= t.x <= x1 and y0 <= t.y <= y1]
    return norm(inner[0].text) if inner else ""


def _feet_guess(labels, page) -> float:
    """PDF points per foot when the scale isn't written: room names are roughly 13 ft apart"""
    pts = [((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) for _, _, r, _ in labels]
    if len(pts) >= 2:
        a = np.array(pts)
        d = np.linalg.norm(a[:, None] - a[None], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1))) / 13
    return min(page.rect.width, page.rect.height) / 80


def _legend_word(text: str) -> bool:
    n = norm(text)
    return bool(LEGEND_TITLE.match(n)) or n.startswith(("SYMBOL", "DESCRIPTION")) or "LEGEND" in n.replace(" ", "")


def _word_paths(drawings: list[dict], lines, matrix) -> set[int]:
    """drawn paths lying inside a word read by OCR (3+ letters): the word's letters"""
    boxes = [r + (-0.5, -0.5, 0.5, 0.5) for t, r in lines if len(norm(t.text).replace(" ", "")) >= 3]
    if not boxes or not drawings:
        return set()
    tree = shapely.STRtree(shapely.box(*np.array([tuple(b) for b in boxes]).T))
    rects = np.array([tuple(d["rect"] * matrix) for d in drawings])
    inside = tree.query(shapely.box(rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]), predicate="within")[0]
    return set(inside.tolist())


def _drop_word_outlines(items: list[geo.Stroke], lines) -> list[geo.Stroke]:
    """Strokes that are the outlines of words read by OCR (3+ letters; short tags like "S1" stay, so a symbol
    with a letter in it keeps its shape)"""
    boxes = [r + (-0.5, -0.5, 0.5, 0.5) for t, r in lines if len(norm(t.text).replace(" ", "")) >= 3]
    if not boxes or not items:
        return items
    tree = shapely.STRtree(shapely.box(*np.array([tuple(b) for b in boxes]).T))
    b = np.array([s.bbox for s in items])
    inside = tree.query(shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]), predicate="within")[0]
    drop = set(inside.tolist())
    return [s for i, s in enumerate(items) if i not in drop]


def _read_page(doc, page_no: int, lines, used_ocr: bool, mapper: SymbolMapper, allow_no_legend: bool,
               given_points: list[PointItem] | None = None, given_outlines: list | None = None) -> dict:
    """Rooms and points of one page. Without a legend (and not allowed) only {"legend": False} comes back.
    given_points: the symbols were already found (a picture: matched as images) - only rooms are read here."""
    page = doc[page_no - 1]
    texts = [t for t, _ in lines]
    drawings = page.get_drawings()
    logs: list[str] = []
    warns: list[str] = []
    if used_ocr:                           # word outlines out before the drawing is broken into strokes
        words = _word_paths(drawings, lines, page.rotation_matrix)
        drawings = [d for i, d in enumerate(drawings) if i not in words]
    items = geo.strokes(drawings, page.rotation_matrix)
    if used_ocr:
        items = _drop_word_outlines(items, lines)
    titles = [(name, t.x, t.y) for t in texts if (name := floor_name(t.text))]
    if len({name for name, _, _ in titles}) == 1:      # the plan title and the title block name the same floor
        titles = titles[:1]
    page_floor = titles[0][0] if len(titles) == 1 else f"Page {page_no}"
    scale = _scale(texts)

    # Step 3 - legend
    shapes, text_symbols, legend_rect, rows = _legend(lines, items, mapper)
    if legend_rect:                       # legend text is not a floor title ("... WEATHER PROOF")
        titles = [tt for tt in titles if not legend_rect.contains(pymupdf.Point(tt[1], tt[2]))]
        if len({name for name, _, _ in titles}) == 1:
            titles = titles[:1]
        page_floor = titles[0][0] if len(titles) == 1 else f"Page {page_no}"
    if given_points is not None:          # (the legend was read from the picture itself)
        allow_no_legend = True
    if not (shapes or text_symbols) and not allow_no_legend:
        return {"page": page_no, "legend": False, "rows": rows}
    lr = tuple(legend_rect) if legend_rect else None
    plan_items = [s for s in items if not (lr and s.inside(lr))]

    # Step 5 - symbols: shapes matching a legend symbol, then short texts ("AC") matching one
    sizes = [s.size for s, _ in shapes]
    max_size = 4 * max(sizes) if sizes else 0.02 * min(page.rect.width, page.rect.height)
    gap = max(0.3, 0.12 * statistics.median(sizes)) if sizes else 0.5
    # strokes in the legend symbols' colours only; all of them when the legend is in black / one colour
    colours = {s.color for s, _ in shapes if s.color}
    dark = {c for c in colours if all(int(v) <= 2 for v in c.split(","))}
    found = [] if given_points is not None else geo.find_symbols(
        plan_items, max_size, gap, colours if colours - dark else None,
        max_line=1.5 * statistics.median(sizes) if sizes else None)
    short_texts = [(k, t) for k, (t, r) in enumerate(lines)
                   if len(t.text.strip()) <= 4 and not (legend_rect and legend_rect.intersects(r))]
    # point types drawn as a filled dot (switches): the area of one dot, per colour
    dot_kinds: dict[str, tuple[str, float]] = {}
    for sym, kind in shapes:
        if sym.filled and sym.fill_area > 0 and sym.color:
            dot_kinds.setdefault(sym.color, (kind, sym.fill_area))
    points: list[PointItem] = list(given_points or [])
    unknown: list[geo.Symbol] = []
    symbol_texts: set[int] = set()        # indexes into lines: texts that are (part of) a symbol
    for sym in found:
        x0, y0, x1, y1 = sym.bbox
        for k, t in ([] if used_ocr else short_texts):   # OCR'd letters stay part of the shape instead
            if x0 <= t.x <= x1 and y0 <= t.y <= y1:
                sym.text = norm(t.text)
                symbol_texts.add(k)
                break
        kind, _ = geo.best_match(sym, shapes)
        if kind:
            points.append(PointItem(kind, sym.x, sym.y))
        elif n := _dots(sym, dot_kinds):
            # touching dots (switches stacked on a wall): one point per dot's worth of filled area
            points += [PointItem(n[0], sym.x, sym.y)] * n[1]
        else:
            unknown.append(sym)
    for k, t in short_texts:
        kind = text_symbols.get(norm(t.text))
        if kind and k not in symbol_texts:
            points.append(PointItem(kind, t.x, t.y))
            symbol_texts.add(k)
    unknown_groups = []
    if not shapes and given_points is None:
        unknown_groups = geo.group_same_shapes(unknown[:1500])
        if unknown_groups:
            warns.append(f"Page {page_no}: {len(unknown_groups)} repeated symbol shape(s) found but not counted (no "
                            "legend says what they are): " + ", ".join(f"shape {chr(65 + i)} x{len(g)}"
                                                                      for i, g in enumerate(unknown_groups[:8])))

    # Step 4 - rooms
    labels = _room_labels(lines, legend_rect, symbol_texts)
    # several drawings on one sheet (floor plans side by side, an elevation, a section, an area plan): each is a
    # floor / section of its own; names in elevations and sections are not rooms
    views = _views(items, lines, page.rect, legend_rect)
    multi = len(views) >= 2 and sum(v["titled"] for v in views) >= 2     # (a title block title alone is not it)
    if multi:
        def view_of(x: float, y: float):
            return next((v for v in views if v["bbox"][0] <= x <= v["bbox"][2] and v["bbox"][1] <= y <= v["bbox"][3]), None)
        plans = 0
        for v in views:
            inside = [lb for lb in labels if view_of((lb[2].x0 + lb[2].x1) / 2, (lb[2].y0 + lb[2].y1) / 2) is v]
            if not v["titled"] and inside:
                v["kind"] = "plan"
            if v["kind"] == "plan" and not v["titled"]:
                plans += 1
                v["name"] = f"Plan {plans} (untitled)"
            if v["kind"] in NOT_PLAN or v["kind"] == "other":
                v["name"] = v["name"] or "Other drawing"
        labels = [lb for lb in labels if (v := view_of((lb[2].x0 + lb[2].x1) / 2, (lb[2].y0 + lb[2].y1) / 2))
                  and v["kind"] == "plan"]
        points = [p for p in points if not ((v := view_of(p.x, p.y)) and v["kind"] in NOT_PLAN)]
        logs.append(f"Page {page_no}: {len(views)} drawings on the sheet - " + ", ".join(
            f"{v['name']} ({'floor plan' if v['kind'] == 'plan' else v['kind'] + (' - not a floor plan' if v['kind'] in NOT_PLAN else '')})"
            for v in views))
    ft = 72 * 12 / scale if scale else _feet_guess(labels, page)
    forbidden = [(t.x, t.y) for t in texts if floor_name(t.text) or LEGEND_TITLE.match(norm(t.text)) or _title_text(t.text)]
    if legend_rect:
        forbidden.append(((legend_rect.x0 + legend_rect.x1) / 2, (legend_rect.y0 + legend_rect.y1) / 2))
    area_range = (12, 6000) if scale else (2, 30000)
    finder = geo.RoomFinder(plan_items, page.rect, ft, area_range)
    building = None if multi else finder.footprint()      # (several drawings: their own areas did that above)
    if building:            # names outside the building's walls (title block, notes, details) are not rooms
        bx0, by0, bx1, by1 = building
        labels = [lb for lb in labels if bx0 <= (lb[2].x0 + lb[2].x1) / 2 <= bx1 and by0 <= (lb[2].y0 + lb[2].y1) / 2 <= by1]
    centres = [((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width / 2) for _, _, r, _ in labels]
    outlines = finder.find(centres, forbidden)
    dims_texts = [(t, r, d) for t, r in lines if (d := parse_dimensions(t.text))]

    # no scale written: work it out from rooms whose size is written ("14'x12'" in a traced outline)
    sqft = (1 / ft ** 2) if scale else None
    if not scale:
        ratios = []
        for (name, kind, rect, h), poly in zip(labels, outlines):
            if poly is not None:
                inside = [d for t, r, d in dims_texts if poly.covers(box(t.x - 0.1, t.y - 0.1, t.x + 0.1, t.y + 0.1))]
                if inside:
                    ratios.append(inside[0][1] / poly.area)
        if ratios:
            sqft = statistics.median(ratios)
            better = 1 / sqft ** 0.5
            if abs(better / ft - 1) > 0.1:      # the guess was off: trace again with the right door width
                ft = better
                outlines = geo.RoomFinder(plan_items, page.rect, ft, area_range).find(centres, forbidden)
            logs.append(f"Page {page_no}: no scale written - worked out from {len(ratios)} room(s) with dimensions")

    # rooms not traced: a room outline found in the picture itself (raster_reader) around the name
    if given_outlines:
        from shapely.geometry import Point, Polygon
        shapes_ = [p for p in (Polygon(o).buffer(0) for o in given_outlines if len(o) >= 3) if not p.is_empty]
        for i, (name, kind, rect, h) in enumerate(labels):
            c = Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            inside = [p for p in shapes_ if p.contains(c)]
            if not inside:
                continue
            best = min(inside, key=lambda p: p.area)
            # (also when the traced outline is a small part of it: furniture around the name, not the room)
            if outlines[i] is None or outlines[i].area < 0.4 * best.area:
                outlines[i] = best

    # rooms not traced: the smallest drawn rectangle around the name
    boxes = []
    for s in plan_items:
        x0, y0, x1, y1 = s.bbox
        if x1 - x0 > 40 and y1 - y0 > 40 and len(s.pts) == 5 and np.allclose(s.pts[0], s.pts[-1]):
            if building and (x0 < building[0] - ft or y0 < building[1] - ft or x1 > building[2] + ft or y1 > building[3] + ft):
                continue                  # bigger than the building (plot limit, sheet frame)
            boxes.append(pymupdf.Rect(s.bbox))
    rooms: list[Room] = []
    traced = 0
    # names that are not in any room, away from the traced rooms (title block, notes): not rooms
    plan_area = None
    polys = [o for o in outlines if o is not None]
    if len(polys) >= 3:
        x0, y0 = min(o.bounds[0] for o in polys), min(o.bounds[1] for o in polys)
        x1, y1 = max(o.bounds[2] for o in polys), max(o.bounds[3] for o in polys)
        pad = 0.05 * max(x1 - x0, y1 - y0)
        plan_area = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    for (name, kind, rect, h), poly in zip(labels, outlines):
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        if poly is None and plan_area is not None and not plan_area.contains(pymupdf.Point(cx, cy)):
            continue
        if poly is not None:
            traced += 1
        else:
            # (a room is far bigger than its name: a rectangle hardly wider than the name is furniture - a table)
            around = [b for b in boxes if b.contains(pymupdf.Point(cx, cy)) and b.width * b.height < page.rect.width * page.rect.height * 0.5
                      and b.width >= 2 * rect.width and b.height >= 3 * rect.height]
            if around:
                b = min(around, key=lambda b: b.width * b.height)
                poly = box(b.x0, b.y0, b.x1, b.y1)
        dims = None
        if poly is not None:
            inside = [(abs(t.x - cx) + abs(t.y - cy), d) for t, r, d in dims_texts if poly.covers(box(t.x - 0.1, t.y - 0.1, t.x + 0.1, t.y + 0.1))]
            dims = min(inside)[1] if inside else None
        if dims is None:
            dims = next((d for t, r, d in dims_texts if abs(t.x - cx) < 6 * h and 0 <= r.y0 - rect.y0 < 3 * h + rect.height), None)
        area = dims[1] if dims else (round(poly.area * sqft, 1) if poly is not None and sqft else None)
        rooms.append(Room(name, poly, cx, cy, kind, dims[0] if dims else None, area))
    # spaces found in a picture with no name read ("ROOM n"): one smaller than any real room is a gap in the
    # furniture or the lettering - left out; the rest numbered again
    if any(re.fullmatch(r"ROOM \d+", r.name) for r in rooms):
        rooms = [r for r in rooms if not (re.fullmatch(r"ROOM \d+", r.name) and r.area_sqft is not None and r.area_sqft < 25)]
        k = 0
        for r in rooms:
            if re.fullmatch(r"ROOM \d+", r.name):
                k += 1
                r.name = f"ROOM {k}"
    # two labels in the same rectangle: keep the room outline only for the first
    seen: set = set()
    for room in rooms:
        if room.poly is not None:
            key = tuple(round(v) for v in room.poly.bounds)
            if key in seen:
                room.poly = None
            seen.add(key)

    # an area plan (the covered area split into parts "A 6'-6" x 7'-0"" ...): each size written in it is an area
    if multi:
        for v in [v for v in views if v["kind"] == "area"]:
            x0, y0, x1, y1 = v["bbox"]
            k = 0
            for t, r, d in dims_texts:
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                    continue
                k += 1
                tag = next((o.text.strip() for o, orr in lines if re.fullmatch(r"[A-Z]\d?", o.text.strip())
                            and 0 < r.y0 - orr.y1 < 4 * r.height and abs((orr.x0 + orr.x1) / 2 - cx) < r.width), None)
                rooms.append(Room(f"Area {tag or k}", None, cx, cy, None, d[0], d[1]))

    counts = Counter(p.kind for p in points)
    logs.append(f"Page {page_no} ({page_floor}): legend {len(shapes) + len(text_symbols)} symbol(s), {len(rooms)} room(s) "
               f"({traced} traced from walls), {len(found)} symbol shape(s) checked; "
               + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no points"))
    sheet = Sheet(default_floor=page_floor, rooms=rooms, points=points,
                  floor_titles=[] if multi else (titles if len(titles) > 1 else []), confidence_penalty=0.1,
                  snap=2.0 * ft, views=[(v["name"], v["bbox"]) for v in views if multi and v["kind"] in ("plan", "area")],
                  debug={"page": page_no - 1, "legend": legend_rect, "unknown": unknown_groups,
                         "views": [{"page": page_no, "name": v["name"], "kind": v["kind"]} for v in views] if multi else []})
    return {"page": page_no, "legend": bool(shapes or text_symbols) or given_points is not None, "rows": rows, "sheet": sheet,
            "log": logs, "warnings": warns}


# pages are read in parallel worker processes (the reading is plain Python, one core per page)
_worker_doc = None
_worker_mapper = None


def _worker_init(data: bytes, mapper: SymbolMapper) -> None:
    global _worker_doc, _worker_mapper
    _worker_doc = pymupdf.open(stream=data, filetype="pdf")
    _worker_mapper = mapper


def _worker(page_no: int, lines, used_ocr: bool, allow: bool, points, outlines) -> dict:
    return _read_page(_worker_doc, page_no, lines, used_ocr, _worker_mapper, allow, points, outlines)


def _read_pages(data: bytes, doc, jobs: list[tuple[int, list, bool]], mapper: SymbolMapper, allow: bool,
                points_of: dict | None = None, outlines_of: dict | None = None) -> list[dict]:
    points_of = points_of or {}
    outlines_of = outlines_of or {}
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1), int(os.environ.get("D2Q_WORKERS", "8")),
                  _workers_for_memory(len(data)))
    done: dict[int, dict] = {}
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        pool = None
        try:
            pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init, initargs=(data, mapper))
            futures = {pool.submit(_worker, n, ln, o, allow, points_of.get(n), outlines_of.get(n)): n for n, ln, o in jobs}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=2)
                for f in finished:
                    done[futures[f]] = f.result()
                # a worker that died (out of memory while starting ...) can leave the pool waiting for ever
                if pending and any(p.exitcode is not None for p in list(getattr(pool, "_processes", {}).values())):
                    break
        except Exception:
            pass                          # no worker processes here (or one failed): the rest one by one
        finally:
            if pool is not None:
                for p in list(getattr(pool, "_processes", {}).values()):
                    if p.exitcode is None:
                        p.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
    return [done[n] if n in done else _read_page(doc, n, ln, o, mapper, allow, points_of.get(n), outlines_of.get(n))
            for n, ln, o in jobs]


def _workers_for_memory(data_size: int) -> int:
    """How many page-reading processes the free memory allows (each holds the PDF and its own Python, ~300 MB)"""
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:
        return 8
    return max(1, int(0.7 * free // (300 * 2 ** 20 + 3 * data_size)))


def read(data: bytes, mapper: SymbolMapper, log: list[str], warnings: list[str],
         pages: list[int] | None = None, ocr_lines: dict[int, list] | None = None,
         points_of: dict[int, list[PointItem]] | None = None,
         outlines_of: dict[int, list] | None = None) -> tuple[dict, list[Sheet], list[dict]]:
    """pages: 1-based page numbers to read (all when None).
    ocr_lines: page number -> text lines already read by OCR (a drawing traced from a picture: raster_reader)
    points_of: page number -> the points already found there (a picture's symbols, matched as images)"""
    doc = pymupdf.open(stream=data, filetype="pdf")
    doc_hash = hashlib.sha1(data).hexdigest()
    project: dict = {}
    legend_all: dict[str, dict] = {}
    ocr_pages: list[int] = []
    if ocr_lines is None:
        log.append(f"Vector PDF, {doc.page_count} page(s) - each page is read as a floor")
    log.append("Step 1/2 - title block and floor titles are read from the page text")
    # 1. the text of every page (OCR for printed pages - it uses all CPU cores itself, so one page at a time)
    texts_of: dict[int, tuple[list, bool]] = {}
    for page_no in [n for n in range(1, doc.page_count + 1) if not pages or n in pages]:
        page = doc[page_no - 1]
        if ocr_lines is not None:
            lines = ocr_lines.get(page_no, [])
            used_ocr = True
        else:
            lines = _lines(page)
            used_ocr = False
        if not lines and not used_ocr and ocr.available():
            if ocr.progress:
                ocr.progress(page_no, doc.page_count)
            lines = ocr.page_lines(page, doc_hash)
            used_ocr = bool(lines)
            if used_ocr:
                ocr_pages.append(page_no)
        texts_of[page_no] = (lines, used_ocr)
        for k, v in title_block([t for t, _ in lines]).items():
            project.setdefault(k, v)
    # 2. rooms and points, pages in parallel. Pages without a legend are read for them only when no page of the
    # drawing has a legend (a drawing set has its legends on the layout sheets; cover, registers, notes, details
    # have none).
    with_heading = [n for n, (ln, _) in texts_of.items() if any(_legend_word(t.text) for t, _ in ln)]
    if points_of is not None:             # a picture: every page with points found is read
        with_heading = sorted(set(with_heading) | {n for n, p in points_of.items() if p})
    results = _read_pages(data, doc, [(n, *texts_of[n]) for n in with_heading], mapper, allow=False, points_of=points_of,
                          outlines_of=outlines_of)
    legend_seen = any(r["legend"] for r in results)
    no_legend = sorted(set(texts_of) - {r["page"] for r in results if r["legend"]})
    if not legend_seen and no_legend:
        results = _read_pages(data, doc, [(n, *texts_of[n]) for n in sorted(texts_of)], mapper, allow=True,
                              points_of=points_of, outlines_of=outlines_of)
    sheets: list[Sheet] = []
    for r in sorted(results, key=lambda r: r["page"]):
        for k, v in r["rows"].items():
            legend_all.setdefault(k, v)
        if "sheet" in r:
            sheets.append(r["sheet"])
            log.extend(r["log"])
            warnings.extend(r["warnings"])
    if ocr_pages:
        warnings.append(f"Page(s) {_pages(ocr_pages)} had no text (printed to PDF) - their words were read from the "
                        "image (OCR); check room names and counts.")
    if no_legend and legend_seen:
        warnings.append(f"Page(s) {_pages(no_legend)}: no legend found - only the title block was read on these pages.")
    elif no_legend:
        warnings.append(f"Page(s) {_pages(no_legend)}: no legend found - symbols on these pages are not counted.")
    log.append("Step 3 - legend: " + (", ".join(f"{l['description']}={l['type']}" for l in legend_all.values()) or "none"))
    return project, sheets, list(legend_all.values())


def _pages(numbers: list[int]) -> str:
    """1, 2, 3, 5 -> "1-3, 5" """
    out, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)
