"""Upload checks: only DXF, or a vector PDF exported from CAD."""
from __future__ import annotations

from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024

SUPPORTED_MESSAGE = (
    "Supported files: DXF (recommended) or a vector PDF exported directly from AutoCAD/CAD software. "
    "Maximum size 50 MB. Not supported: DWG files (please save as DXF), scanned PDFs, photos or images (JPG/PNG), "
    "hand-drawn plans, or password-protected PDFs. For best results, the drawing should include a legend and room names."
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif", ".webp"}


class RejectedFile(Exception):
    """The file can't be read; the message tells the user why and what to upload instead"""


def check(file_name: str, data: bytes) -> str:
    """'dxf' or 'pdf' - raises RejectedFile with the reason otherwise"""
    ext = Path(file_name or "").suffix.lower()
    if not data:
        raise RejectedFile("The file is empty.")
    if len(data) > MAX_BYTES:
        raise RejectedFile(f"The file is {len(data) / 1024 / 1024:.1f} MB - the maximum is 50 MB.")
    if ext == ".dwg" or data[:4] in (b"AC10", b"AC1."):
        raise RejectedFile("DWG files can't be read. In AutoCAD use Save As -> DXF and upload the DXF file.")
    if ext in IMAGE_EXT or data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
        raise RejectedFile("Photos and images (JPG/PNG ...) can't be read. Upload the DXF, or the PDF exported directly from AutoCAD/CAD software.")
    if ext == ".pdf" or data[:5] == b"%PDF-":
        _check_pdf(data)
        return "pdf"
    if ext == ".dxf":
        head = data[:4096].decode("latin-1", errors="ignore")
        if "SECTION" not in head and not data.startswith(b"AutoCAD Binary DXF"):
            raise RejectedFile("This file is not a valid DXF drawing. Save it from AutoCAD as DXF and upload it again.")
        return "dxf"
    raise RejectedFile(f"\"{ext or 'this'}\" files are not supported. " + SUPPORTED_MESSAGE)


def _check_pdf(data: bytes) -> None:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise RejectedFile("This PDF is damaged and can't be opened.")
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise RejectedFile("This PDF is password-protected. Upload a copy without a password.")
        if doc.page_count == 0:
            raise RejectedFile("This PDF has no pages.")
        has_text = has_vectors = False
        for page in doc:
            if page.get_text("text").strip():
                has_text = True
            # count line pieces, not paths: CAD exports often put thousands of lines in one path
            if not has_vectors and sum(len(d["items"]) for d in page.get_drawings()) >= 20:
                has_vectors = True
            if has_text and has_vectors:
                return
        if not has_text and not has_vectors:
            raise RejectedFile("This PDF appears to be scanned (no selectable text or vector lines). "
                               "Please upload the original CAD-exported PDF or a DXF file.")
        if not has_text:
            from .ocr import available
            if has_vectors and available():
                return          # printed to PDF: the words are read from the page image (OCR)
            raise RejectedFile("This PDF has the drawing lines but no selectable text - it was printed to PDF, so every "
                               "word became lines and the room names and legend can't be read. Install the OCR reader "
                               "(pip install rapidocr-onnxruntime), or upload the DXF (in AutoCAD: Save As -> DXF).")
        if not has_vectors:
            raise RejectedFile("This PDF has no vector drawing (it looks like a picture of a plan). "
                               "Please upload the original CAD-exported PDF or a DXF file.")
