"""Synthetic drawings with known counts (python -m tests.fixtures writes them to tests/data)."""
from __future__ import annotations

from pathlib import Path

import ezdxf
import pymupdf

DATA = Path(__file__).parent / "data"

# ground floor (x 0..40) and first floor (x 100..130); feet
ROOMS = {
    "MASTER BED ROOM": ((0, 0, 14, 14), "14'x14'", "Ground Floor"),
    "LOUNGE": ((14, 0, 30, 20), "16'x20'", "Ground Floor"),
    "KITCHEN": ((30, 0, 40, 12), "10'x12'", "Ground Floor"),
    "BED ROOM": ((100, 0, 112, 14), None, "First Floor"),
    "GUEST ROOM": ((112, 0, 124, 12), "12'x12'", "First Floor"),
}
# legend symbols (neutral block names: only the legend says what they are)
LEGEND = [("E01", "LIGHT POINT", "light"), ("E02", "CEILING FAN", "fan"), ("E03", "5A SOCKET", "socket_5A"),
          ("E04", "15A POWER SOCKET", "socket_15A"), ("E05", "AC POINT", "ac_point"), ("E06", "CURTAIN", "curtain")]
EXPECTED = {
    "MASTER BED ROOM": {"light": 6, "fan": 1, "socket_5A": 4, "socket_15A": 1, "ac_point": 1, "curtain": 2},
    "LOUNGE": {"light": 8, "fan": 2, "socket_5A": 6, "ac_point": 1},
    "KITCHEN": {"light": 4, "exhaust_fan": 1, "socket_5A": 3, "socket_15A": 2},
    "BED ROOM": {"light": 3, "fan": 1, "ac_point": 1},
    "GUEST ROOM": {"light": 2, "fan": 1},
}
UNASSIGNED = {"light": 1}
BLOCK_OF = {t: b for b, _, t in LEGEND} | {"exhaust_fan": "EXHAUST_FAN"}


def _spots(rect, n):
    x0, y0, x1, y1 = rect
    cols = max(1, int(n ** 0.5 + 0.999))
    rows = (n + cols - 1) // cols
    for i in range(n):
        r, c = divmod(i, cols)
        yield x0 + (x1 - x0) * (c + 1) / (cols + 1), y0 + (y1 - y0) * (r + 1) / (rows + 2)


def make_dxf(path: Path = DATA / "sample.dxf") -> Path:
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 2   # feet
    for name in list(BLOCK_OF.values()) + ["SOFA", "XYZ1"]:
        blk = doc.blocks.new(name)
        blk.add_circle((0, 0), 0.3)
    msp = doc.modelspace()
    doc.layers.add("A-ROOM")
    for name, (rect, dims, _) in ROOMS.items():
        x0, y0, x1, y1 = rect
        msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True, dxfattribs={"layer": "A-ROOM"})
        msp.add_text(name, dxfattribs={"height": 0.6, "insert": (x0 + 1, y1 - 1.5)})
        if dims:
            msp.add_text(dims, dxfattribs={"height": 0.4, "insert": (x0 + 1, y1 - 2.5)})
        for kind, n in EXPECTED[name].items():
            for x, y in _spots(rect, n):
                msp.add_blockref(BLOCK_OF[kind], (x + (0.2 if kind == "fan" else 0), y))
    # a plot boundary around the ground floor (no name: must not become a room)
    msp.add_lwpolyline([(-2, -2), (42, -2), (42, 22), (-2, 22)], close=True)
    msp.add_blockref("E01", (41, 21))          # a light outside every room -> Unassigned
    msp.add_blockref("SOFA", (20, 10))         # furniture: ignored
    msp.add_blockref("XYZ1", (5, 5))           # unknown: warned about
    msp.add_text("GROUND FLOOR PLAN", dxfattribs={"height": 1, "insert": (12, -5)})
    msp.add_text("FIRST FLOOR PLAN", dxfattribs={"height": 1, "insert": (106, -5)})
    # legend
    msp.add_text("LEGEND", dxfattribs={"height": 1, "insert": (250, 40)})
    for i, (block, desc, _) in enumerate(LEGEND):
        y = 37 - i * 2.5
        msp.add_blockref(block, (251, y))
        msp.add_text(desc, dxfattribs={"height": 0.8, "insert": (253, y - 0.3)})
    # title block
    for i, t in enumerate(["PROJECT: Khan Villa", "CLIENT: Mr. Ahmed Khan", "DRAWING NO: E-101", "SCALE: 1:100",
                           "DATE: 01-10-2026", "ARCHITECT: Studio A", "PLOT: 10 MARLA"]):
        msp.add_text(t, dxfattribs={"height": 0.8, "insert": (250, 10 - i * 1.5)})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


def make_pdf(path: Path = DATA / "sample.pdf") -> Path:
    """Two pages (one per floor), 1 ft = 12 pt (scale 1:72); light = circle, fan = square, 5A socket = triangle,
    15A = filled triangle, AC = the text "AC", curtain = diamond; exhaust fan = small filled circle"""
    doc = pymupdf.open()

    def symbol(page, kind, x, y):
        s = page.new_shape()
        if kind == "light":
            s.draw_circle((x, y), 4)
            s.finish(color=(0, 0, 0), width=0.8)
        elif kind == "exhaust_fan":
            s.draw_circle((x, y), 3)
            s.finish(color=(0, 0, 0), fill=(0, 0, 0))
        elif kind == "fan":
            s.draw_rect(pymupdf.Rect(x - 4, y - 4, x + 4, y + 4))
            s.finish(color=(0, 0, 0), width=0.8)
        elif kind in ("socket_5A", "socket_15A"):
            s.draw_polyline([(x - 4, y + 3), (x + 4, y + 3), (x, y - 4), (x - 4, y + 3)])
            s.finish(color=(0, 0, 0), fill=(0, 0, 0) if kind == "socket_15A" else None, closePath=True)
        elif kind == "curtain":
            s.draw_polyline([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y), (x, y - 5)])
            s.finish(color=(0, 0, 0), closePath=True)
        elif kind == "ac_point":
            page.insert_text((x - 6, y + 3), "AC", fontsize=7)
            return
        s.commit()

    legend = [("light", "LIGHT POINT"), ("fan", "CEILING FAN"), ("socket_5A", "5A SOCKET"), ("socket_15A", "15A POWER SOCKET"),
              ("ac_point", "AC POINT"), ("curtain", "CURTAIN"), ("exhaust_fan", "EXHAUST FAN")]
    for floor, title in (("Ground Floor", "GROUND FLOOR PLAN"), ("First Floor", "FIRST FLOOR PLAN")):
        page = doc.new_page(width=842, height=595)
        page.insert_text((60, 40), title, fontsize=14)
        for x, t in ((60, "SCALE 1:72"), (200, "PROJECT: Khan Villa"), (380, "PLOT: 10 MARLA")):
            page.insert_text((x, 575), t, fontsize=8)
        for name, (rect, dims, fl) in ROOMS.items():
            if fl != floor:
                continue
            x0, y0, x1, y1 = rect
            dx = 0 if floor == "Ground Floor" else 100     # 1 ft = 12 pt = 1 inch at 1:72
            R = pymupdf.Rect(60 + (x0 - dx) * 12, 80 + y0 * 12, 60 + (x1 - dx) * 12, 80 + y1 * 12)
            s = page.new_shape(); s.draw_rect(R); s.finish(color=(0, 0, 0), width=1.5); s.commit()
            page.insert_text((R.x0 + 6, R.y0 + 14), name, fontsize=8)
            if dims:
                page.insert_text((R.x0 + 6, R.y0 + 24), dims, fontsize=7)
            counts = EXPECTED[name]
            spots = list(_spots((R.x0 + 4, R.y0 + 30, R.x1 - 4, R.y1 - 4), sum(counts.values())))
            k = 0
            for kind, n in counts.items():
                for _ in range(n):
                    symbol(page, kind, *spots[k]); k += 1
        if floor == "Ground Floor":
            symbol(page, "light", 580, 480)     # outside every room
        page.insert_text((640, 80), "LEGEND", fontsize=10)
        for i, (kind, desc) in enumerate(legend):
            y = 100 + i * 18
            symbol(page, kind, 650, y)
            page.insert_text((665, y + 3), desc, fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


if __name__ == "__main__":
    print(make_dxf(), make_pdf())
