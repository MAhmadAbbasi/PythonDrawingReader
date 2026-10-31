"""Reads a drawing into floors -> areas -> electrical point counts (see extract())."""
from __future__ import annotations

from . import dxf_reader, pdf_reader
from .common import SymbolMapper, assemble, totals
from .validator import RejectedFile, SUPPORTED_MESSAGE, check

__all__ = ["extract", "extract_with_sheets", "RejectedFile", "SUPPORTED_MESSAGE"]


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
    reader = dxf_reader if kind == "dxf" else pdf_reader
    try:
        if kind == "pdf":
            project, sheets, legend = reader.read(data, mapper, log, warnings, pages)
        else:
            project, sheets, legend = reader.read(data, mapper, log, warnings)
    except RejectedFile:
        raise
    except Exception as exc:
        raise RejectedFile(f"The drawing could not be read: {exc}")
    floors = assemble(sheets, log, warnings)
    log.append("Step 6/7 - points outside rooms are in \"Unassigned\"; every area has a confidence score")
    return {
        "fileName": file_name,
        "fileType": kind,
        "project": project,
        "floors": floors,
        "legend": legend,
        "totals": totals(floors),
        "warnings": warnings,
        "log": log,
        "pointTypes": mapper.types_list,
    }, sheets
