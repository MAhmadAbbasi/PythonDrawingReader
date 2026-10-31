"""Text for PDFs whose words became lines (printed to PDF through a printer driver, e.g. "PScript5 / Distiller"):
the page is rendered as an image and read with RapidOCR, which runs offline on onnxruntime - no cloud service.

Reading is slow (about half a minute per A3 sheet on a normal PC), so results are cached on disk by the page's
content: the same drawing is only read once. Set D2Q_OCR_CACHE to choose the cache folder."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pymupdf

from .common import Text

DPI = 200
TILE, OVERLAP = 720, 120         # pixels; words cut by a tile edge are read whole in the next tile
MIN_CONFIDENCE = 0.6
_engine = None
progress = None          # optional callback(page_no, page_count), called before a page is read


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _cache_dir() -> Path:
    d = Path(os.environ.get("D2Q_OCR_CACHE") or Path(tempfile.gettempdir()) / "drawing2quote-ocr")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_image(img: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    # 1. find where the words are, tile by tile (small tiles: the detector shrinks big images and loses small text)
    boxes = []
    h, w = img.shape[:2]
    for y in range(0, h, TILE - OVERLAP):
        for x in range(0, w, TILE - OVERLAP):
            tile = np.ascontiguousarray(img[y:y + TILE, x:x + TILE])
            if (tile.min(axis=2) < 200).mean() < 0.002:      # (nearly) empty paper
                continue
            quads, _ = _engine(tile, use_rec=False, use_cls=False)
            for quad in quads or []:
                q = np.array(quad, float) + (x, y)
                boxes.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
    if not boxes:
        return []
    # 2. tiles overlap: the same word found twice, or a long line cut by a tile edge -> one box
    boxes = _merge_boxes(boxes)
    # 3. read every word once, all in one batch (reading is the slow part)
    quads = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in boxes], dtype=np.float32)
    crops = _engine.get_crop_img_list(img, quads)
    texts, _ = _engine.text_rec(crops)
    return [(str(r[0]), float(r[1]), tuple(b)) for r, b in zip(texts, boxes)]


def _merge_boxes(boxes: list[list[float]]) -> list[list[float]]:
    """Boxes on the same text line that overlap become one (union); a box inside a bigger one is dropped"""
    b = sorted(boxes, key=lambda r: r[0])
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        for r in b:
            for k in out:
                ix = min(r[2], k[2]) - max(r[0], k[0])
                iy = min(r[3], k[3]) - max(r[1], k[1])
                hr, hk = r[3] - r[1], k[3] - k[1]
                if ix > 0 and iy > 0.6 * min(hr, hk) and max(hr, hk) < 1.6 * min(hr, hk):
                    k[0], k[1], k[2], k[3] = min(k[0], r[0]), min(k[1], r[1]), max(k[2], r[2]), max(k[3], r[3])
                    merged = True
                    break
                area = max((r[2] - r[0]) * (r[3] - r[1]), 1e-6)
                if ix > 0 and iy > 0 and ix * iy > 0.7 * area:
                    merged = True                # mostly inside a box already kept
                    break
            else:
                out.append(list(r))
        b = out
    return b


def _dedupe(words: list[tuple[str, float, tuple]]) -> list[tuple[str, float, tuple]]:
    """Tiles overlap, so a word can be read twice or in pieces: of overlapping boxes keep the longest text"""
    words = sorted(words, key=lambda w: (-len(w[0].strip()), -w[1]))
    kept: list[tuple[str, float, tuple]] = []
    for text, conf, b in words:
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        dup = False
        for _, _, k in kept:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            if ix * iy > 0.5 * min(area, (k[2] - k[0]) * (k[3] - k[1])):
                dup = True
                break
        if not dup:
            kept.append((text, conf, b))
    return kept


def page_lines(page: pymupdf.Page, doc_hash: str) -> list[tuple[Text, pymupdf.Rect]]:
    """Text lines read from the page image, in the coordinates of the page as shown (rotation applied)"""
    key = hashlib.sha1(f"v2:{doc_hash}:{page.number}:{DPI}:{TILE}".encode()).hexdigest()
    cache = _cache_dir() / f"{key}.json"
    words = None
    if cache.exists():
        try:
            words = [(t, c, tuple(b)) for t, c, b in json.loads(cache.read_text(encoding="utf8"))]
        except Exception:
            words = None
    if words is None:
        pix = page.get_pixmap(dpi=DPI, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        words = _dedupe([w for w in _read_image(img) if w[1] >= MIN_CONFIDENCE and w[0].strip()])
        cache.write_text(json.dumps(words), encoding="utf8")
    zoom = DPI / 72
    out = []
    for text, _, (x0, y0, x1, y1) in words:
        r = pymupdf.Rect(x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom)
        out.append((Text(text.strip(), (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.height * 0.8), r))
    return out
