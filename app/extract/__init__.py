"""Reads a drawing into floors -> areas -> electrical point counts (see extract())."""
from __future__ import annotations

import sys

# pymupdf loads its optional AI layout add-on (pymupdf-layout, an ONNX model) on import when it is installed. It is
# not used here, and loading it in every page-reading worker process ran out of memory and left the pool stuck.
# (None in sys.modules: pymupdf's "import pymupdf.layout" fails with ImportError, which it skips)
if "pymupdf" not in sys.modules:
    sys.modules.setdefault("pymupdf.layout", None)

from . import dxf_reader, pdf_reader, raster_reader  # noqa: E402
from .common import SymbolMapper, assemble, totals
from .validator import RejectedFile, SUPPORTED_MESSAGE, check

__all__ = ["extract", "extract_with_sheets", "RejectedFile", "SUPPORTED_MESSAGE", "NOT_ELECTRICAL"]

NOT_ELECTRICAL = ("Not an electrical drawing: it has no electrical legend or points (an architectural layout). "
                  "The solution is suggested from the rooms and areas found only - lights, switches and sockets are "
                  "estimated per room.")


def _looks_like_plan(sheets) -> bool:
    """A picture is a floor plan when something of one was read in it: a room name, a floor title, symbols, or
    (names too small to read in a blurred photo) several rooms traced between walls. A screenshot or a table has
    none of these - at most a space or two between its lines."""
    for s in sheets:
        if s.points or (s.floor_titles or []) or (getattr(s, "views", None) or []) \
                or not s.default_floor.startswith("Page "):
            return True
        if any(r.kind and r.kind != "other" for r in s.rooms):
            return True
        if sum(1 for r in s.rooms if r.poly is not None) >= 4:
            return True
    return False


def extract(file_name: str, data: bytes, mapper: SymbolMapper | None = None) -> dict:
    """The drawing's project info, floors with their areas and point counts, legend, warnings and a step log.
    Raises RejectedFile when the file can't be read (with the reason)."""
    return extract_with_sheets(file_name, data, mapper)[0]


def extract_with_sheets(file_name: str, data: bytes, mapper: SymbolMapper | None = None,
                        pages: list[int] | None = None) -> tuple[dict, list]:
    """extract(), plus the rooms and points as read on each sheet (for overlay.draw).
    pages: PDF page numbers to read (1-based; all when None)"""
    kind = check(file_name, data)
    mapper = mapper or SymbolMapper()
    log: list[str] = []
    warnings: list[str] = []
    try:
        if kind == "pdf":
            project, sheets, legend = pdf_reader.read(data, mapper, log, warnings, pages)
        elif kind in ("scan", "image"):
            # traced into a drawing first; the overlay is drawn on that traced drawing
            project, sheets, legend, traced = raster_reader.read(data, kind, mapper, log, warnings, pages)
            for s in sheets:
                (s.debug if s.debug is not None else {}).setdefault("pdf", traced)
        else:
            project, sheets, legend = dxf_reader.read(data, mapper, log, warnings)
    except RejectedFile:
        raise
    except Exception as exc:
        raise RejectedFile(f"The drawing could not be read: {exc}")
    if kind in ("scan", "image") and not _looks_like_plan(sheets):
        raise RejectedFile("This picture doesn't look like a floor plan: no room names, floor titles or electrical "
                           "symbols were found in it. Send a clear photo or scan of the drawing.")
    floors = assemble(sheets, log, warnings)
    log.append("Step 6/7 - points outside rooms are in \"Unassigned\"; every area has a confidence score")
    sums = totals(floors)
    # every drawing found on the sheets (floor plans, area plans, elevations, sections ...): nothing left out unsaid
    sections = []
    for s in sheets:
        for v in (s.debug or {}).get("views", []):
            floor = next((f for f in floors if f["name"] == v["name"]), None)
            sections.append({**v, "rooms": len([a for a in floor["areas"] if a["name"] != "Unassigned"]) if floor else 0,
                             "areaSqft": floor["areaSqft"] if floor else None})
    others = [f"{v['name']} (page {v['page']})" for v in sections if v["kind"] not in ("plan", "area")]
    if others:
        warnings.append("Also on the sheets, not floor plans (no rooms taken from them): " + ", ".join(others) + ".")
    # an architectural layout (no electrical legend, no points): the solution comes from the rooms alone
    electrical = bool(legend) or sums.get("points", 0) > 0
    if not electrical:
        # (the "no legend found" note would only repeat it; repeated shapes stay listed - symbols drawn without a
        # legend can be told apart there)
        warnings[:] = [w for w in warnings if "no legend found" not in w]
        warnings.insert(0, NOT_ELECTRICAL)
    return {
        "fileName": file_name,
        "fileType": kind,
        "project": project,
        "floors": floors,
        "legend": legend,
        "totals": sums,
        "sections": sections,
        "electrical": electrical,
        "warnings": warnings,
        "log": log,
        "pointTypes": mapper.types_list,
    }, sheets
