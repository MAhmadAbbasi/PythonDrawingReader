"""Geometry of a vector PDF page, used by pdf_reader:

* strokes()      - every drawn path broken into simple strokes (lines, curves as short lines, rectangles)
* find_symbols() - small strokes that touch are grouped into one symbol; each symbol gets a shape description
                   that ignores rotation and position, so a socket turned 90 degrees on a wall still matches
                   the legend symbol
* RoomFinder     - room outlines from the wall lines: door openings are closed by extending wall ends across
                   the gap, and the enclosed space around each room name is the room

Only numpy + shapely; no AI, no image processing."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon, box

DASHED = re.compile(r"\[\s*[\d.]*[1-9]")      # "[3 2] 0" is dashed, "[] 0" is solid
MAX_WALL_SEGMENTS = 30000                     # a line set bigger than this is not tried for rooms (too slow)
ROOM_TIME_LIMIT = 45.0                         # seconds of room tracing per page, then the rest are names only
MATCH_LIMIT = 0.5                              # shape distance under which a symbol is the legend symbol
SAME_SHAPE = 0.3                               # two unknown symbols with the same shape


# ── strokes ──
@dataclass
class Stroke:
    pts: np.ndarray                            # (n, 2) polyline
    width: float                               # line weight; filled shapes rank as the heaviest
    filled: bool
    dashed: bool
    curve: bool
    bbox: tuple[float, float, float, float]
    color: str = ""                            # "r,g,b" in tenths (fill colour for filled shapes)
    path_size: float = 0.0                     # size of the drawn path this stroke is part of
    chain_size: float = 0.0                    # size of the run of strokes joined end to end it belongs to
    path_id: int = -1                          # which drawn path it came from

    @property
    def piece_size(self) -> float:
        """a curve flattened into tiny lines (as printer drivers do) counts as the whole curve"""
        return max(self.size, self.chain_size)

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])

    def inside(self, r) -> bool:
        return r[0] <= self.bbox[0] and r[1] <= self.bbox[1] and self.bbox[2] <= r[2] and self.bbox[3] <= r[3]


def _colour(rgb) -> str:
    return ",".join(str(round(c * 10)) for c in rgb[:3]) if rgb else ""


def _bezier(p0, p1, p2, p3, n: int = 8) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def strokes(drawings: list[dict], matrix=None) -> list[Stroke]:
    """matrix: turns PyMuPDF's unrotated coordinates into the page as shown (page.rotation_matrix)"""
    out: list[Stroke] = []
    rot = np.array([[matrix.a, matrix.b], [matrix.c, matrix.d]]) if matrix is not None else None
    shift = np.array([matrix.e, matrix.f]) if matrix is not None else None
    widths = [d.get("width") or 0.0 for d in drawings]
    heaviest = (max(widths) if widths else 0.0) + 1.0
    for path_id, d in enumerate(drawings):
        fill = d.get("fill")
        filled = fill is not None and min(fill) < 0.9 and (d.get("fill_opacity") or 1.0) > 0.2
        stroked = "s" in (d.get("type") or "s")
        dashed = stroked and bool(DASHED.search(d.get("dashes") or ""))
        width = heaviest if filled else (d.get("width") or 0.0)
        color = _colour(fill if filled else d.get("color"))
        path_size = max(d["rect"].width, d["rect"].height)
        first = len(out)
        chains: list[int] = []                 # chain number of each stroke of this path
        last_end = None
        for item in d["items"]:
            op, curve = item[0], False
            if op == "l":
                pts = np.array([tuple(item[1]), tuple(item[2])], float)
            elif op == "c":
                pts = _bezier(*(np.array(tuple(p), float) for p in item[1:5]))
                curve = True
            elif op == "re":
                r = item[1]
                pts = np.array([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)], float)
            elif op == "qu":
                q = item[1]
                pts = np.array([tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll), tuple(q.ul)], float)
            else:
                continue
            if rot is not None:
                pts = pts @ rot + shift
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            joined = last_end is not None and op in ("l", "c") and abs(pts[0] - last_end).max() < 0.01
            chains.append(chains[-1] if joined and chains else (chains[-1] + 1 if chains else 0))
            last_end = pts[-1] if op in ("l", "c") else None
            out.append(Stroke(pts, width, filled, dashed, curve, (lo[0], lo[1], hi[0], hi[1]), color, path_size, path_id=path_id))
        if chains:
            box_of: dict[int, list[float]] = {}
            for c, m in zip(chains, out[first:]):
                bb = box_of.setdefault(c, list(m.bbox))
                bb[0], bb[1] = min(bb[0], m.bbox[0]), min(bb[1], m.bbox[1])
                bb[2], bb[3] = max(bb[2], m.bbox[2]), max(bb[3], m.bbox[3])
            for c, m in zip(chains, out[first:]):
                bb = box_of[c]
                m.chain_size = max(bb[2] - bb[0], bb[3] - bb[1])
    return out


def _segments(items: list[Stroke]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """start points, end points, width, is-curve, is-filled of every straight piece"""
    p0, p1, w, c, f = [], [], [], [], []
    for s in items:
        n = len(s.pts) - 1
        if n < 1:
            continue
        p0.append(s.pts[:-1]); p1.append(s.pts[1:])
        w.append(np.full(n, s.width)); c.append(np.full(n, s.curve)); f.append(np.full(n, s.filled))
    if not p0:
        e = np.zeros((0, 2))
        return e, e, np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
    return np.vstack(p0), np.vstack(p1), np.concatenate(w), np.concatenate(c), np.concatenate(f)


# ── symbols ──
@dataclass
class Symbol:
    bbox: tuple[float, float, float, float]
    x: float
    y: float
    r: float                                   # furthest ink from the centre
    length: float                              # total ink length / r
    radial: np.ndarray                         # how the ink is spread from the centre outwards (6 bins)
    dirs: np.ndarray                           # line directions (12 bins over 180 degrees)
    curve: float                               # share of the ink that is curved
    filled: bool
    w: float
    h: float
    text: str = ""                             # a short text inside the symbol ("S", "TV" ...)
    color: str = ""                            # the colour most of the ink has
    fill_area: float = 0.0                     # area covered by filled parts (dots), in pt²

    @property
    def size(self) -> float:
        return max(self.w, self.h)


def describe(items: list[Stroke]) -> Symbol | None:
    p0, p1, _, curve, filled = _segments(items)
    # the same line drawn twice (common in CAD exports) counts once
    a, b = np.round(p0, 2), np.round(p1, 2)
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    key = np.hstack([np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)])
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    p0, p1, curve, filled = p0[first], p1[first], curve[first], filled[first]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    total = lengths.sum()
    if total <= 0:
        return None
    lo = np.minimum(p0, p1).min(axis=0)
    hi = np.maximum(p0, p1).max(axis=0)
    w, h = hi - lo
    # sample the ink evenly along its length
    step = max(w, h) / 24 or 1.0
    k = np.maximum(1, np.ceil(lengths / step)).astype(int)
    seg = np.repeat(np.arange(len(lengths)), k)
    t = (np.concatenate([np.arange(n) for n in k]) + 0.5) / k[seg]
    pts = p0[seg] + (p1[seg] - p0[seg]) * t[:, None]
    wts = lengths[seg] / k[seg]
    centre = (wts[:, None] * pts).sum(axis=0) / wts.sum()
    ends = np.vstack([p0, p1])
    r = max(float(np.linalg.norm(ends - centre, axis=1).max()), 1e-6)
    radial = np.histogram(np.linalg.norm(pts - centre, axis=1) / r, bins=6, range=(0, 1.0001), weights=wts)[0] / total
    # directions, softly split between the two nearest bins so a 1-degree turn changes little
    ang = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0]) % math.pi
    pos = ang / (math.pi / 12)
    low = np.floor(pos).astype(int) % 12
    frac = pos - np.floor(pos)
    dirs = np.zeros(12)
    np.add.at(dirs, low, lengths * (1 - frac))
    np.add.at(dirs, (low + 1) % 12, lengths * frac)
    fill_area = 0.0
    filled_items = [st for st in items if st.filled]
    if filled_items:
        # printer drivers paint a filled dot as many thin slivers: measure the area they cover together
        shapes_ = [shapely.Polygon(st.pts) if len(st.pts) >= 4 else shapely.LineString(st.pts) for st in filled_items]
        fill_area = float(shapely.union_all(shapely.buffer(shapes_, 0.2, quad_segs=1)).area)
    ink: dict[str, float] = {}
    for st in items:
        ink[st.color] = ink.get(st.color, 0.0) + float(np.linalg.norm(np.diff(st.pts, axis=0), axis=1).sum())
    return Symbol(fill_area=fill_area, color=max(ink, key=ink.get) if ink else "", bbox=(lo[0], lo[1], hi[0], hi[1]), x=float(centre[0]), y=float(centre[1]), r=r, length=total / r,
                  radial=radial, dirs=dirs / total, curve=float(lengths[curve].sum() / total),
                  filled=bool(lengths[filled].sum() / total > 0.5), w=float(w), h=float(h))


def _rolls(sym: Symbol) -> np.ndarray:
    """the 12 rotations of a symbol's direction histogram (kept on the symbol: legend symbols are compared often)"""
    r = getattr(sym, "_rolls", None)
    if r is None:
        r = np.stack([np.roll(sym.dirs, k) for k in range(12)])
        object.__setattr__(sym, "_rolls", r)
    return r


def shape_distance(a: Symbol, b: Symbol) -> float:
    """0 for the same shape at any rotation; about 0.5 and more for different shapes"""
    if a.filled != b.filled or a.text != b.text:
        return math.inf
    # plans often draw symbols at another size than the legend (a fan at its real sweep), so size counts little
    d = 0.4 * abs(math.log(a.r / b.r))
    if a.color and b.color and a.color != b.color:
        d += 0.4
    d += abs(a.length - b.length) / max(a.length, b.length)
    d += 0.5 * np.abs(a.radial - b.radial).sum()
    d += abs(a.curve - b.curve)
    d += 0.5 * float(np.abs(_rolls(b) - a.dirs).sum(axis=1).min())
    # the outline box, allowing quarter turns (a square and the same square standing on a corner differ)
    aw, ah = max(a.w, 0.05 * a.r), max(a.h, 0.05 * a.r)
    bw, bh = max(b.w, 0.05 * b.r), max(b.h, 0.05 * b.r)
    d += 0.25 * min(abs(math.log(aw / bw)) + abs(math.log(ah / bh)), abs(math.log(aw / bh)) + abs(math.log(ah / bw)))
    return d


def _run_extent(items: list[Stroke]) -> list[float]:
    """For each stroke, the size of the run of strokes joined end to end through plain bends (exactly two
    pieces meeting). A wire drawn as hundreds of tiny separate pieces is one long run and is left out; a symbol
    touching a wire meets it at a junction (three pieces), so the symbol stays separate."""
    if not items:
        return []
    def vid(pt) -> tuple[int, int]:
        return (int(round(pt[0] * 20)), int(round(pt[1] * 20)))          # 0.05 pt grid
    edge_of: dict[tuple, int] = {}
    stroke_edge: list[int] = []
    for st in items:
        a, b = vid(st.pts[0]), vid(st.pts[-1])
        key = (min(a, b), max(a, b))
        stroke_edge.append(edge_of.setdefault(key, len(edge_of)) if a != b else -1)
    edges = list(edge_of)
    at: dict[tuple, list[int]] = {}
    for e, (a, b) in enumerate(edges):
        at.setdefault(a, []).append(e)
        at.setdefault(b, []).append(e)
    parent = list(range(len(edges)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for es in at.values():
        if len(es) == 2:
            ra, rb = root(es[0]), root(es[1])
            if ra != rb:
                parent[ra] = rb
    box_of: dict[int, list[float]] = {}
    for st, e in zip(items, stroke_edge):
        if e < 0:
            continue
        r = root(e)
        bb = box_of.setdefault(r, list(st.bbox))
        bb[0], bb[1] = min(bb[0], st.bbox[0]), min(bb[1], st.bbox[1])
        bb[2], bb[3] = max(bb[2], st.bbox[2]), max(bb[3], st.bbox[3])
    out = []
    for st, e in zip(items, stroke_edge):
        if e < 0:
            out.append(st.size)
        else:
            bb = box_of[root(e)]
            out.append(max(bb[2] - bb[0], bb[3] - bb[1]))
    return out


def find_symbols(items: list[Stroke], max_size: float, gap: float, colours: set[str] | None = None,
                 max_line: float | None = None) -> list[Symbol]:
    """Groups strokes no bigger than max_size that touch (within gap) into symbols. colours: only strokes in these
    colours (the legend symbols' colours - leaves out grey furniture and walls that touch the symbols).
    max_line: a single straight line longer than this is wiring, not part of a symbol"""
    small = [s for s in items if s.piece_size <= max_size and not s.dashed and (not colours or s.color in colours)]
    if max_line:
        small = [s for s in small if len(s.pts) > 2 or s.size <= max_line]
    ext = _run_extent(small)
    small = [s for s, e in zip(small, ext) if e <= max_size]
    if not small:
        return []
    b = np.array([s.bbox for s in small])
    boxes = shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3])
    grown = shapely.box(b[:, 0] - gap, b[:, 1] - gap, b[:, 2] + gap, b[:, 3] + gap)
    pairs = shapely.STRtree(boxes).query(grown, predicate="intersects")
    parent = list(range(len(small)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs.T:
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(small)):
        groups.setdefault(root(i), []).append(i)
    out = []
    for members in groups.values():
        g = b[members]
        if max(g[:, 2].max() - g[:, 0].min(), g[:, 3].max() - g[:, 1].min()) > 1.25 * max_size:
            continue        # a chain of small strokes (hatching, stairs ...), not one symbol
        for part in _split_at_dots([small[i] for i in members]):
            sym = describe(part)
            if sym is not None:
                out.append(sym)
    return out


def _split_at_dots(items: list[Stroke]) -> list[list[Stroke]]:
    """Symbols drawn touching each other (switches stacked on a wall, a legend row of switch types) each have
    their own filled dot: two or more separate dots of about the same size -> one symbol per dot, each taking the
    lines nearest to it"""
    dots = [s for s in items if s.filled]
    if len(dots) < 2:
        return [items]
    b = np.array([s.bbox for s in dots])
    pairs = shapely.STRtree(shapely.box(*b.T)).query(shapely.box(*b.T), predicate="intersects")
    parent = list(range(len(dots)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs.T:
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            parent[ri] = rj
    blobs: dict[int, list[int]] = {}
    for i in range(len(dots)):
        blobs.setdefault(root(i), []).append(i)
    if len(blobs) < 2:
        return [items]
    boxes = [(b[m][:, 0].min(), b[m][:, 1].min(), b[m][:, 2].max(), b[m][:, 3].max()) for m in blobs.values()]
    sizes = [max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes]
    if min(sizes) < 0.6 * max(sizes):
        return [items]                       # one dot with a smaller mark, not a row of symbols
    centres = np.array([((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes])
    parts: list[list[Stroke]] = [[] for _ in boxes]
    for s in items:
        c = np.array(((s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2))
        parts[int(np.argmin(np.linalg.norm(centres - c, axis=1)))].append(s)
    return [p for p in parts if p]


def best_match(sym: Symbol, legend: list[tuple[Symbol, str]]) -> tuple[str | None, float]:
    best, dist = None, math.inf
    for ref, kind in legend:
        d = shape_distance(sym, ref)
        if d < dist:
            best, dist = kind, d
    return (best, dist) if dist < MATCH_LIMIT else (None, dist)


def group_same_shapes(symbols: list[Symbol], min_count: int = 3) -> list[list[Symbol]]:
    """Unknown symbols with the same shape, most frequent first (used when there is no legend)"""
    groups: list[list[Symbol]] = []
    for s in symbols[:5000]:
        g = next((g for g in groups if shape_distance(s, g[0]) < SAME_SHAPE), None)
        if g is None:
            groups.append([s])
        else:
            g.append(s)
    return sorted((g for g in groups if len(g) >= min_count), key=len, reverse=True)


# ── rooms ──
def _closed_runs(members: list[Stroke]) -> list[np.ndarray]:
    """the outlines (point lists) of a filled path: its pieces joined end to end"""
    runs, cur = [], []
    for st in members:
        if len(st.pts) >= 4 and abs(st.pts[0] - st.pts[-1]).max() < 0.01:   # a rectangle / closed piece
            runs.append(st.pts)
            continue
        if cur and abs(cur[-1][-1] - st.pts[0]).max() >= 0.01:
            runs.append(np.vstack(cur)); cur = []
        cur.append(st.pts if not cur else st.pts[1:])
    if cur:
        runs.append(np.vstack(cur))
    return runs


class RoomFinder:
    """Rooms are the enclosed spaces around the room names.

    Wall sets are tried in turn: the walls' own colour (the colour of the big filled shapes - walls drawn solid),
    then line weights from the heaviest down (walls are drawn heavier than furniture, hatching and dimensions),
    then everything. Door openings are closed two ways: wall lines are extended up to a door width until they
    meet another line, and solid walls are thickened by half a door width (and the room grown back by as much).
    A space is accepted as a room when it has exactly one room name in it, does not reach the edge of the page
    (the outside of the building), and its area is believable."""

    def __init__(self, items: list[Stroke], page_rect, ft: float, area_range: tuple[float, float]):
        self.page = box(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
        self.page_bounds = (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
        self.ft = ft                                       # PDF points per foot
        self.area_range = area_range                       # sq ft
        # walls are parts of big paths: symbols, letters, hatching and loose furniture bits are left out
        walls = [s for s in items if not s.dashed and s.path_size >= 3 * ft]
        # solid walls: each filled path is one shape (printer drivers paint them as thin triangles of 2-point
        # pieces); its pieces are long themselves - a filled symbol sharing a big path with others is not a wall
        by_path: dict[int, list[Stroke]] = {}
        for st in walls:
            if st.filled:
                by_path.setdefault(st.path_id, []).append(st)
        solid: list[Stroke] = []
        for members in by_path.values():
            for run in _closed_runs(members):
                lo, hi = run.min(axis=0), run.max(axis=0)
                if len(run) >= 3 and max(hi - lo) >= 2 * ft:
                    solid.append(Stroke(run, members[0].width, True, False, False, (lo[0], lo[1], hi[0], hi[1]),
                                        members[0].color, members[0].path_size, path_id=members[0].path_id))
        solid_paths = {st.path_id for st in solid}
        lines = [s for s in walls if not (s.filled and s.path_id in solid_paths)]
        self.p0, self.p1, self.w, self.curve, _ = _segments(lines)
        seg_colour = np.array([c for s in lines for c in [s.color] * (len(s.pts) - 1)], dtype=object)
        self.solid = solid
        self.eps = max(0.1 * ft, 0.3)
        # wall sets: (lines to use, solid shapes to use)
        sets: list[tuple[np.ndarray, list[Stroke]]] = []
        if solid:
            colours: dict[str, int] = {}
            for st in solid:
                colours[st.color] = colours.get(st.color, 0) + 1
            wall_colour = max(colours, key=colours.get)
            sets.append((seg_colour == wall_colour, [st for st in solid if st.color == wall_colour]))
        widths = sorted({round(float(v), 2) for v in self.w}, reverse=True)
        for t in (widths[:3] if len(widths) > 1 else []):
            sets.append((self.w >= t - 1e-6, solid))
        sets.append((np.ones(len(self.w), bool), solid))
        self.sets = sets
        self.gaps = [1.5 * ft, 4.0 * ft]                   # door openings up to 4 ft are closed
        self._cache: dict = {}
        self._mass: dict = {}

    def footprint(self) -> tuple[float, float, float, float] | None:
        """The box around the building: the biggest connected mass of solid walls (None without solid walls)"""
        if not self.sets or not self.sets[0][1]:
            return None
        polys = shapely.buffer(np.array([shapely.Polygon(st.pts) for st in self.sets[0][1]], dtype=object), 0)
        mass = shapely.union_all(shapely.buffer(polys, 2 * self.ft, quad_segs=1))
        # a building can fall apart into a few wall masses (open or double-height spaces between them): all big ones
        parts = sorted(shapely.get_parts(mass), key=lambda g: -g.area)
        keep = [g for g in parts if g.area >= 0.15 * parts[0].area]
        return shapely.union_all(keep).bounds

    def _spaces(self, k: int, gap: float):
        key = (k, gap)
        if key in self._cache:
            return self._cache[key]
        sel, solid = self.sets[k]
        p0, p1, curve = self.p0[sel], self.p1[sel], self.curve[sel]
        if len(p0) + len(solid) == 0 or len(p0) > MAX_WALL_SEGMENTS or len(solid) > MAX_WALL_SEGMENTS:
            self._cache[key] = None
            return None
        geoms = []
        if len(p0):
            lines = shapely.linestrings(np.stack([p0, p1], axis=1))
            lengths = np.linalg.norm(p1 - p0, axis=1)
            # extend straight wall ends across openings
            ext = np.nonzero(~curve & (lengths >= 1.5 * self.ft))[0]
            extra = []
            if len(ext):
                u = (p1[ext] - p0[ext]) / lengths[ext, None]
                ends = np.vstack([p1[ext], p0[ext]])
                dirs = np.vstack([u, -u])
                owner = np.concatenate([ext, ext])
                start = ends + dirs * 0.05
                rays = shapely.linestrings(np.stack([start, ends + dirs * gap], axis=1))
                ri, si = shapely.STRtree(lines).query(rays, predicate="intersects")
                keep = si != owner[ri]
                ri, si = ri[keep], si[keep]
                if len(ri):
                    hit = shapely.line_locate_point(rays[ri], shapely.centroid(shapely.intersection(rays[ri], lines[si])))
                    best = np.full(len(rays), np.inf)
                    np.minimum.at(best, ri, hit)
                    found = np.isfinite(best)
                    if found.any():
                        tips = start[found] + dirs[found] * (best[found, None] + self.eps)
                        extra = list(shapely.linestrings(np.stack([ends[found], tips], axis=1)))
            # each line thickened on its own, then joined: the same shape, far faster than thickening them as one
            # shape when many lines overlap (46 s -> 1.4 s on a busy sheet)
            pieces = shapely.buffer(np.concatenate([lines, np.array(extra, dtype=object)]) if extra else lines,
                                    self.eps, quad_segs=1, cap_style="square")
            geoms.append(shapely.union_all(pieces))
        mass = None
        if solid:
            # solid walls (often painted as many thin slivers): their area, thickened to close door openings
            mass = self._mass.get(k)
            if mass is None:
                polys = shapely.buffer(np.array([shapely.Polygon(st.pts) for st in solid], dtype=object), 0)
                mass = self._mass[k] = shapely.union_all(shapely.buffer(polys, self.eps, quad_segs=1))
            geoms.append(mass.buffer(gap / 2, quad_segs=2))
        walls = shapely.union_all(geoms)
        parts = shapely.get_parts(self.page.difference(walls))
        result = (parts, shapely.STRtree(parts), mass, gap / 2 if solid else 0.0)
        self._cache[key] = result
        return result

    def _ok(self, part, i: int, xs: np.ndarray, ys: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> bool:
        x0, y0, x1, y1 = part.bounds
        px0, py0, px1, py1 = self.page_bounds
        if x0 <= px0 + 1 or y0 <= py0 + 1 or x1 >= px1 - 1 or y1 >= py1 - 1:
            return False                                   # the outside of the building
        fp = self._footprint
        if fp and (x0 < fp[0] - self.ft or y0 < fp[1] - self.ft or x1 > fp[2] + self.ft or y1 > fp[3] + self.ft):
            return False                                   # reaches past the building's walls
        sqft = part.area / self.ft ** 2
        if not (self.area_range[0] <= sqft <= self.area_range[1]) or part.area > 0.5 * self.page.area:
            return False
        inside = shapely.contains_xy(part, xs, ys)
        inside[i] = False
        if inside.any():
            return False                                   # two rooms joined through an open doorway
        return not (len(fx) and shapely.contains_xy(part, fx, fy).any())

    def find(self, labels: list[tuple[float, float, float]], forbidden: list[tuple[float, float]]) -> list[Polygon | None]:
        """A room outline per label (x, y, half the text width); None when no enclosed space fits.
        The label is tested at its centre and towards both ends, since a name is often written across a line
        (a door leaf, furniture); of the spaces it touches the largest acceptable one wins."""
        xs = np.array([p[0] for p in labels], float)
        ys = np.array([p[1] for p in labels], float)
        probes = [shapely.points([(x, y), (x - 0.6 * hw, y), (x + 0.6 * hw, y)]) for x, y, hw in labels]
        fx = np.array([p[0] for p in forbidden], float)
        fy = np.array([p[1] for p in forbidden], float)
        out: list[Polygon | None] = [None] * len(labels)
        todo = set(range(len(labels)))
        self._footprint = self.footprint()
        deadline = time.monotonic() + ROOM_TIME_LIMIT
        self.timed_out = False
        for k in range(len(self.sets)):
            for gap in self.gaps:
                if todo and time.monotonic() > deadline:
                    self.timed_out = True
                    return out
                if not todo:
                    return out
                spaces = self._spaces(k, gap)
                if spaces is None:
                    continue
                parts, tree, mass, grow = spaces
                for i in sorted(todo):
                    hits = set(tree.query(probes[i], predicate="within")[1].tolist())
                    if not hits:              # the whole label sits on lines: the spaces right next to it
                        hits = set(tree.query(probes[i], predicate="dwithin", distance=self.ft)[1].tolist())
                    for j in sorted(hits, key=lambda j: -parts[j].area):
                        if self._ok(parts[j], i, xs, ys, fx, fy):
                            room = Polygon(parts[j].exterior).buffer(self.eps + grow, join_style="mitre")
                            if mass is not None:
                                room = room.difference(mass)           # grown back up to the wall faces
                            if room.geom_type != "Polygon":
                                room = max(shapely.get_parts(room), key=lambda g: g.area)
                            out[i] = room
                            todo.discard(i)
                            break
        return out
