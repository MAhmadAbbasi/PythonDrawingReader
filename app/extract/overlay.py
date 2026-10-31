"""A copy of the PDF with what was read drawn on top, to check the reading by eye:
room outlines (coloured, with the name), every counted symbol (a box with its type), the legend area, and -
on pages without a legend - the repeated symbol shapes that were found but not counted (grey, lettered)."""
from __future__ import annotations

import colorsys
import zlib

import pymupdf

from .common import Sheet

SHORT = {"light": "L", "fan": "F", "exhaust_fan": "EF", "socket_5A": "5A", "socket_15A": "15A", "switch": "SW",
         "ac_point": "AC", "tv_point": "TV", "data_point": "D", "curtain": "C", "geyser": "G", "db_panel": "DB"}


def _colour(name: str, light: float = 0.45) -> tuple[float, float, float]:
    return colorsys.hls_to_rgb((zlib.crc32(name.encode()) % 360) / 360, light, 0.85)


def draw(data: bytes, sheets: list[Sheet]) -> bytes:
    doc = pymupdf.open(stream=data, filetype="pdf")
    for sheet in sheets:
        if "page" not in sheet.debug:
            continue
        page = doc[sheet.debug["page"]]
        m = page.derotation_matrix          # readings are in the page as shown; drawing is in the unrotated page
        s = page.new_shape()
        for i, room in enumerate(sheet.rooms):
            col = _colour(room.name + str(i), 0.5)
            if room.poly is not None:
                pts = [pymupdf.Point(x, y) * m for x, y in room.poly.exterior.coords]
                s.draw_polyline(pts)
                s.finish(color=col, fill=col, fill_opacity=0.18, width=1.2, closePath=True)
            else:
                s.draw_circle(pymupdf.Point(room.x, room.y) * m, 6)
                s.finish(color=col, width=1.2, dashes="[2 2] 0")
        for p in sheet.points:
            col = _colour(p.kind, 0.4)
            s.draw_rect(pymupdf.Rect(p.x - 5, p.y - 5, p.x + 5, p.y + 5) * m)
            s.finish(color=col, width=0.8)
        legend = sheet.debug.get("legend")
        if legend:
            s.draw_rect(legend * m)
            s.finish(color=(0, 0.4, 1), width=1, dashes="[4 2] 0")
        for g, group in enumerate(sheet.debug.get("unknown") or []):
            for sym in group:
                s.draw_rect(pymupdf.Rect(sym.bbox) * m)
                s.finish(color=(0.5, 0.5, 0.5), width=0.6, dashes="[1 1] 0")
        s.commit()
        for i, room in enumerate(sheet.rooms):
            label = room.name + (f"  {room.area_sqft:g} sq ft" if room.area_sqft else "")
            page.insert_text(pymupdf.Point(room.x - 2 * len(label), room.y - 8) * m, label, fontsize=6,
                             color=_colour(room.name + str(i), 0.3), rotate=page.rotation)
        for p in sheet.points:
            page.insert_text(pymupdf.Point(p.x + 5.5, p.y - 5) * m, SHORT.get(p.kind, p.kind), fontsize=4.5,
                             color=_colour(p.kind, 0.35), rotate=page.rotation)
        for g, group in enumerate(sheet.debug.get("unknown") or []):
            for sym in group:
                page.insert_text(pymupdf.Point(sym.bbox[2] + 1, sym.bbox[1]) * m, chr(65 + g % 26), fontsize=4.5,
                                 color=(0.4, 0.4, 0.4), rotate=page.rotation)
    return doc.tobytes(garbage=3, deflate=True)
