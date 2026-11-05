"""Upload checks: DXF, a PDF of a CAD drawing (exported, or printed - then read with OCR), or a scanned PDF or
picture of a drawing (read from the image)."""
from __future__ import annotations

from pathlib import Path

MAX_MB = 200     # keep the same in the POS: DrawingService.MaxBytes, web.config, drawingService.ts
MAX_BYTES = MAX_MB * 1024 * 1024

# keep the same wording as READ_QUALITY in the POS frontend (drawingService.ts)
SUPPORTED_MESSAGE = (
    "Best: DXF or CAD-exported PDF (exact) · Good: CAD PDF without text (OCR, slower) · Fair: scan or clear "
    "JPG/PNG (check counts) · Low: blurry photo/screenshot (rooms & sizes estimated). "
    f"Cannot read: DWG (save as DXF), locked PDF, HEIC, under 300 px, over {MAX_MB} MB. "
    "Points need a legend; send WhatsApp images as Document."
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"II*\x00", b"MM\x00*", b"BM", b"GIF8", b"RIFF")


class RejectedFile(Exception):
    """The file can't be read; the message tells the user why and what to upload instead"""


def check(file_name: str, data: bytes) -> str:
    """'dxf', 'pdf' (CAD PDF), 'scan' (scanned PDF) or 'image' - raises RejectedFile with the reason otherwise"""
    ext = Path(file_name or "").suffix.lower()
    if not data:
        raise RejectedFile("The file is empty.")
    if len(data) > MAX_BYTES:
        raise RejectedFile(f"The file is {len(data) / 1024 / 1024:.1f} MB - the maximum is {MAX_MB} MB.")
    if ext == ".dwg" or data[:4] in (b"AC10", b"AC1."):
        raise RejectedFile("DWG files can't be read. In AutoCAD use Save As -> DXF and upload the DXF file.")
    if ext in (".heic", ".heif"):
        raise RejectedFile("HEIC photos can't be read. On the phone choose \"Most Compatible\" (JPG) or send it as JPG.")
    if ext in IMAGE_EXT or data.startswith(IMAGE_MAGIC):
        _check_image(data)
        return "image"
    if ext == ".pdf" or data[:5] == b"%PDF-":
        return _check_pdf(data)
    if ext == ".dxf":
        head = data[:4096].decode("latin-1", errors="ignore")
        if "SECTION" not in head and not data.startswith(b"AutoCAD Binary DXF"):
            raise RejectedFile("This file is not a valid DXF drawing. Save it from AutoCAD as DXF and upload it again.")
        return "dxf"
    raise RejectedFile(f"\"{ext or 'this'}\" files are not supported. " + SUPPORTED_MESSAGE)


def _need_ocr() -> None:
    from .ocr import available
    if not available():
        raise RejectedFile("Scanned drawings and pictures need the OCR reader: pip install rapidocr-onnxruntime. "
                           "Or upload the DXF (in AutoCAD: Save As -> DXF).")


def _check_image(data: bytes) -> None:
    from PIL import Image
    import io
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        raise RejectedFile("This picture is damaged or not a picture format that can be read (use JPG, PNG or TIFF).")
    if min(w, h) < 300:
        raise RejectedFile(f"This picture is too small ({w} x {h} pixels) to read the room names and symbols. "
                           "Scan or photograph the drawing at a higher resolution (at least 2000 pixels wide).")
    _need_ocr()


def _check_pdf(data: bytes) -> str:
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
                return "pdf"
        if not has_vectors:
            # scanned: the pages are pictures (a searchable scan has a text layer too) - read from the image
            if not any(page.get_images() for page in doc):
                raise RejectedFile("This PDF has no drawing in it (no lines and no scanned pages).")
            _need_ocr()
            return "scan"
        if not has_text:
            from .ocr import available
            if available():
                return "pdf"    # printed to PDF: the words are read from the page image (OCR)
            raise RejectedFile("This PDF has the drawing lines but no selectable text - it was printed to PDF, so every "
                               "word became lines and the room names and legend can't be read. Install the OCR reader "
                               "(pip install rapidocr-onnxruntime), or upload the DXF (in AutoCAD: Save As -> DXF).")
        return "pdf"
