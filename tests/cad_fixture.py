"""A CAD-style plan PDF with known counts (python -m tests.cad_fixture writes it to tests/data).

Like real exports: double-line walls with door openings (some with a door swing, some without), furniture,
floor hatching, dashed wiring, a dimension line, symbols made of several strokes (all light circles in one path,
all their crosses in another), sockets turned to follow the walls, a room name on two lines - and no rectangles
drawn around the rooms."""
from __future__ import annotations

import math
from pathlib import Path

import pymupdf

DATA = Path(__file__).parent / "data"

ROOMS = {   # name lines, (x0, y0, x1, y1) in feet, dimension text
    "MASTER BED ROOM": (["MASTER", "BED ROOM"], (0, 0, 14, 16), "14'x16'"),
    "LOUNGE": (["LOUNGE"], (14, 0, 28, 16), "14'x16'"),
    "KITCHEN": (["KITCHEN"], (28, 0, 40, 10), None),
    "BATH": (["BATH"], (28, 10, 40, 16), None),
    "PASSAGE": (["PASSAGE"], (0, 16, 40, 21), None),
    "BED ROOM": (["BED ROOM"], (0, 21, 14, 32), "14'x11'"),
    "GUEST ROOM": (["GUEST ROOM"], (14, 21, 26, 32), None),
    "DINING": (["DINING"], (26, 21, 40, 32), "14'x11'"),
}
# symbols in the middle of each room
CENTRE = {
    "MASTER BED ROOM": {"light": 4, "fan": 1, "curtain": 1},
    "LOUNGE": {"light": 6, "fan": 2},
    "KITCHEN": {"light": 2, "exhaust_fan": 1},
    "BATH": {"light": 1, "exhaust_fan": 1},
    "PASSAGE": {"light": 3},
    "BED ROOM": {"light": 3, "fan": 1},
    "GUEST ROOM": {"light": 2, "fan": 1},
    "DINING": {"light": 3, "fan": 1},
}
# symbols on the walls: (kind, wall side, position along the wall 0..1)
ON_WALLS = {
    "MASTER BED ROOM": [("socket_5A", "W", .3), ("socket_5A", "W", .7), ("socket_5A", "N", .5), ("switch", "S", .7),
                        ("ac_point", "N", .2), ("tv_point", "E", .5)],
    "LOUNGE": [("socket_5A", "N", .3), ("socket_5A", "N", .7), ("socket_5A", "W", .3), ("socket_5A", "E", .8),
               ("socket_15A", "N", .9), ("switch", "S", .1), ("switch", "S", .8), ("tv_point", "W", .7), ("ac_point", "N", .5)],
    "KITCHEN": [("socket_15A", "N", .3), ("socket_15A", "N", .6), ("socket_5A", "E", .4), ("socket_5A", "E", .7), ("switch", "W", .8)],
    "BATH": [("switch", "W", .5)],
    "PASSAGE": [("switch", "N", .05)],
    "BED ROOM": [("socket_5A", "W", .4), ("socket_5A", "S", .5), ("switch", "N", .15), ("ac_point", "S", .8)],
    "GUEST ROOM": [("socket_5A", "S", .3), ("socket_5A", "E", .5), ("switch", "N", .7)],
    "DINING": [("socket_5A", "S", .3), ("socket_5A", "E", .5)],
}
EXPECTED: dict[str, dict[str, int]] = {}
for _room in ROOMS:
    _c = dict(CENTRE[_room])
    for _kind, _, _ in ON_WALLS[_room]:
        _c[_kind] = _c.get(_kind, 0) + 1
    EXPECTED[_room] = dict(sorted(_c.items()))
UNASSIGNED = {"light": 1}
# wall centre lines (feet) and their openings along the wall (from, to, door swing drawn?)
WALLS = [
    ((0, 0), (40, 0), []), ((0, 32), (40, 32), []), ((0, 0), (0, 32), []),
    ((40, 0), (40, 32), [(17, 20, True)]),                                     # main entrance
    ((14, 0), (14, 16), []), ((14, 21), (14, 32), []),
    ((28, 0), (28, 16), [(3, 6, True)]),                                       # kitchen from the lounge
    ((28, 10), (40, 10), []),
    ((0, 16), (40, 16), [(5, 8, True), (18, 21.5, False), (31, 33.5, True)]),  # master, lounge arch, bath
    ((0, 21), (40, 21), [(5, 8, True), (17, 20, True), (30, 33, False)]),      # bed, guest, dining (open)
    ((26, 21), (26, 32), []),
]
WALL = 0.75
FT = 72 * 12 / 100                       # PDF points per foot at 1:100
LEGEND = [("light", "LIGHT POINT"), ("fan", "CEILING FAN"), ("exhaust_fan", "EXHAUST FAN"), ("socket_5A", "5A SOCKET"),
          ("socket_15A", "15A POWER SOCKET"), ("switch", "SWITCH BOARD"), ("tv_point", "TV POINT"),
          ("ac_point", "AC POINT"), ("curtain", "CURTAIN")]


class _Pen:
    """The symbol shapes; strokes of one kind share a path, as CAD exports do"""

    def __init__(self, page):
        self.page = page
        self.circles, self.crosses, self.other, self.filled = (page.new_shape() for _ in range(4))

    def draw(self, kind, c, vertical=False, turn=0.0):
        o = self.other
        if kind == "light":                      # circle with a cross
            self.circles.draw_circle(c, 4)
            for ang in (turn + math.pi / 4, turn + 3 * math.pi / 4):
                d = pymupdf.Point(math.cos(ang), math.sin(ang)) * 4
                self.crosses.draw_line(c - d, c + d)
        elif kind == "fan":                      # circle with three blades
            o.draw_circle(c, 5)
            for ang in (0, 2 * math.pi / 3, 4 * math.pi / 3):
                o.draw_line(c, c + pymupdf.Point(math.cos(ang), math.sin(ang)) * 5)
        elif kind == "exhaust_fan":              # circle with a square
            o.draw_circle(c, 4)
            o.draw_rect(pymupdf.Rect(c.x - 2, c.y - 2, c.x + 2, c.y + 2))
        elif kind == "curtain":                  # zigzag
            o.draw_polyline([c + (-6 + 2.4 * i, -2.5 if i % 2 else 2.5) for i in range(6)])
        elif kind in ("socket_5A", "socket_15A"):   # box with two pins / filled box, along the wall
            w, h = (2.5, 4.5) if vertical else (4.5, 2.5)
            (self.filled if kind == "socket_15A" else o).draw_rect(pymupdf.Rect(c.x - w, c.y - h, c.x + w, c.y + h))
            if kind == "socket_5A":
                for t in (-1.5, 1.5):
                    o.draw_line(c + ((-1.5, t) if vertical else (t, -1.5)), c + ((1.5, t) if vertical else (t, 1.5)))
        elif kind in ("switch", "tv_point"):     # box with letters
            letters = "S" if kind == "switch" else "TV"
            o.draw_rect(pymupdf.Rect(c.x - 4, c.y - 4, c.x + 4, c.y + 4))
            self.page.insert_text((c.x - 1.6 * len(letters), c.y + 2), letters, fontsize=5)
        elif kind == "ac_point":                 # just the letters
            self.page.insert_text((c.x - 4, c.y + 2), "AC", fontsize=6)

    def commit(self):
        for s in (self.circles, self.crosses, self.other):
            s.finish(color=(0, 0, 0), width=0.6, closePath=False)
            s.commit()
        self.filled.finish(color=(0, 0, 0), fill=(0, 0, 0), width=0.6, closePath=False)
        self.filled.commit()


def _walls(page, P):
    walls, doors = page.new_shape(), page.new_shape()
    for a, b, openings in WALLS:
        horizontal = a[1] == b[1]
        lo, hi = (a[0], b[0]) if horizontal else (a[1], b[1])
        pieces, start = [], lo - WALL / 2
        for o0, o1, swing in sorted(openings):
            pieces.append((start, o0))
            start = o1
            if swing:      # door leaf at the first jamb, quarter swing to the other
                if horizontal:
                    hinge, leaf, other = P(o0, a[1]), P(o0, a[1] + (o1 - o0)), P(o1, a[1])
                else:
                    hinge, leaf, other = P(a[0], o0), P(a[0] - (o1 - o0), o0), P(a[0], o1)
                doors.draw_line(hinge, leaf)
                doors.draw_bezier(leaf, leaf + (other - hinge) * 0.5523, other + (leaf - hinge) * 0.5523, other)
        pieces.append((start, hi + WALL / 2))
        for s, e in pieces:   # both faces and the end caps of each piece, as separate lines
            if horizontal:
                pts = [P(s, a[1] - WALL / 2), P(e, a[1] - WALL / 2), P(e, a[1] + WALL / 2), P(s, a[1] + WALL / 2)]
            else:
                pts = [P(a[0] - WALL / 2, s), P(a[0] + WALL / 2, s), P(a[0] + WALL / 2, e), P(a[0] - WALL / 2, e)]
            for i in range(4):
                walls.draw_line(pts[i], pts[(i + 1) % 4])
    walls.finish(color=(0, 0, 0), width=1.4, closePath=False)
    doors.finish(color=(0, 0, 0), width=0.5, closePath=False)
    walls.commit()
    doors.commit()


def _spots(name, x0, y0, x1, y1, n, label_x):
    if name == "PASSAGE":
        return [(x, 18.5) for x in (15, 25, 35)]
    if name == "BATH":
        return [(31, 13.2), (36, 13.2)]
    top = y0 + 5
    cols = max(1, math.ceil(math.sqrt(n * (x1 - x0) / (y1 - top))))
    rows = math.ceil(n / cols)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((x0 + (x1 - x0) * (c + 1) / (cols + 1), top + (y1 - 1.5 - top) * (r + 0.5) / rows))
    return out


def make_cad_pdf(path: Path = DATA / "sample_cad.pdf", scale_text: bool = True, legend: bool = True) -> Path:
    OX, OY = 90, 90

    def P(x, y):
        return pymupdf.Point(OX + x * FT, OY + y * FT)

    doc = pymupdf.open()
    page = doc.new_page(width=1191, height=842)          # A3 landscape
    _walls(page, P)
    s = page.new_shape()                                 # furniture: a bed and a sofa
    s.draw_rect(pymupdf.Rect(P(4, 6), P(10, 13)))
    s.draw_rect(pymupdf.Rect(P(4, 6), P(10, 7.5)))
    s.draw_rect(pymupdf.Rect(P(16, 12), P(25, 15)))
    s.finish(color=(0, 0, 0), width=0.3); s.commit()
    s = page.new_shape()                                 # kitchen floor tiles
    for i in range(1, 12):
        s.draw_line(P(28 + i, 0.5), P(28 + i, 9.5))
    for j in range(1, 10):
        s.draw_line(P(28.5, j), P(39.5, j))
    s.finish(color=(0.5, 0.5, 0.5), width=0.2); s.commit()
    s = page.new_shape()                                 # dashed wiring
    s.draw_bezier(P(9.8, 15.2), P(9, 11), P(6, 9), P(4.7, 4.5))
    s.finish(color=(0, 0, 0), width=0.5, dashes="[3 2] 0"); s.commit()
    s = page.new_shape()                                 # overall dimension line with ticks
    s.draw_line(P(0, -3), P(40, -3))
    for x in (0, 40):
        s.draw_line(P(x, -3) + (-2, 2), P(x, -3) + (2, -2))
    s.finish(color=(0, 0, 0), width=0.25); s.commit()
    page.insert_text(P(18, -3.5), "40'-0\"", fontsize=6)

    pen = _Pen(page)
    n_light = 0
    for name, (label, (x0, y0, x1, y1), dims) in ROOMS.items():
        lx = (x0 + x1) / 2 if name != "PASSAGE" else x0 + 5
        ly = y0 + (2.6 if name == "PASSAGE" else 2.2)
        for i, line in enumerate(label):
            page.insert_text(P(lx, ly) + (-len(line) * 2, i * 8), line, fontsize=7)
        if dims:
            page.insert_text(P(lx, ly) + (-14, len(label) * 8), dims, fontsize=6)
        kinds = [k for k, n in CENTRE[name].items() for _ in range(n)]
        for kind, (x, y) in zip(kinds, _spots(name, x0, y0, x1, y1, len(kinds), lx)):
            if kind == "light":
                n_light += 1
            pen.draw(kind, P(x, y), turn=math.pi / 4 if n_light % 5 == 0 else 0.0)   # every 5th light drawn as "+"
        for kind, side, t in ON_WALLS[name]:
            face = WALL / 2
            c, vertical = {"N": (P(x0 + (x1 - x0) * t, y0 + face), False), "S": (P(x0 + (x1 - x0) * t, y1 - face), False),
                           "W": (P(x0 + face, y0 + (y1 - y0) * t), True), "E": (P(x1 - face, y0 + (y1 - y0) * t), True)}[side]
            if kind in ("switch", "tv_point", "ac_point"):       # drawn just inside the wall
                c = c + {"N": (0, 6), "S": (0, -6), "W": (6, 0), "E": (-6, 0)}[side]
            pen.draw(kind, c, vertical)
    pen.draw("light", P(43.5, 26))                       # a garden light outside the building -> Unassigned
    page.insert_text(P(12, 36), "GROUND FLOOR PLAN", fontsize=12)
    if legend:
        page.insert_text((760, 100), "ELECTRICAL LEGEND", fontsize=10)
        for i, (kind, desc) in enumerate(LEGEND):
            y = 122 + i * 18
            pen.draw(kind, pymupdf.Point(772, y))
            page.insert_text((790, y + 3), desc, fontsize=7)
    pen.commit()
    for i, t in enumerate(["PROJECT: Test House", "CLIENT: Mr. Test", "SCALE 1:100" if scale_text else "SCALE: AS SHOWN",
                           "DRAWING NO: E-201"]):
        page.insert_text((900, 720 + i * 14), t, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


if __name__ == "__main__":
    print(make_cad_pdf())
