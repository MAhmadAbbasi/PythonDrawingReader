"""Scanned PDFs and pictures of a drawing (JPG, PNG, TIFF ...).

A picture has no lines or text to read, only pixels, so it is first traced into a drawing the PDF reader
understands, and then read exactly like a CAD PDF that was printed to PDF:
  1. ink: dark or coloured pixels on the paper (uneven lighting of photos is evened out first)
  2. words: read with OCR (cached, like printed PDFs); words of 3+ characters are wiped from the ink
  3. thick ink (solid walls, filled dots) -> filled shapes (walls cut into pieces without holes, so the rooms
     inside a wall ring stay open); thin ink -> centre lines, traced into polylines that end where lines meet,
     like the separate lines of a CAD drawing (a wire ends at the symbol it feeds)
  4. that traced drawing (one PDF page per picture page) is read by pdf_reader: legend, symbols, rooms, areas"""
from __future__ import annotations

import io
import re

import cv2
import numpy as np
import pymupdf
import shapely
from shapely.geometry import Polygon, box

from collections import Counter

from . import ocr, pdf_geometry as geo, pdf_reader
from .common import PointItem, Sheet, SymbolMapper, Text, norm, parse_dimensions, room_type

LONG_SIDE = (3000, 7000)       # pixels: smaller pictures are enlarged (small words), bigger ones reduced (speed)
NOMINAL_DPI = 200              # a picture without a trustworthy resolution: its "paper" size is unknown


def _rgb(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, :3])


def _fit(img: np.ndarray) -> tuple[np.ndarray, float]:
    """the picture at a size OCR and tracing work well at, and how much it was scaled"""
    long_side = max(img.shape[:2])
    s = 1.0
    if long_side < LONG_SIDE[0]:
        s = LONG_SIDE[0] / long_side
    elif long_side > LONG_SIDE[1]:
        s = LONG_SIDE[1] / long_side
    if abs(s - 1) > 0.02:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return img, s


def pictures(data: bytes, kind: str, pages: list[int] | None) -> list[tuple[int, np.ndarray, float, bool, bool]]:
    """(page number, RGB image, PDF points per pixel, is the paper size known, was it enlarged) for each page"""
    out = []
    if kind == "scan":
        doc = pymupdf.open(stream=data, filetype="pdf")
        for page in doc:
            if pages and page.number + 1 not in pages:
                continue
            # the resolution the page was scanned at (its biggest picture), kept between 150 and 300 dpi
            native = 200.0
            infos = [i for i in page.get_image_info() if i.get("width")]
            if infos:
                big = max(infos, key=lambda i: (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
                w_in = (big["bbox"][2] - big["bbox"][0]) / 72
                if w_in > 0:
                    native = big["width"] / w_in
            dpi = min(max(native, 150), 300)
            dpi = max(min(dpi, LONG_SIDE[1] * 72 / max(page.rect.width, page.rect.height)), 72)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            img, s = _fit(_rgb(img))
            out.append((page.number + 1, img, 72 / (dpi * s), True, s > 1.02))
        return out
    from PIL import Image, ImageOps, ImageSequence
    im = Image.open(io.BytesIO(data))
    dpi = im.info.get("dpi")
    dpi = float(dpi[0]) if dpi and dpi[0] else 0.0
    known = dpi >= 100 and round(dpi) not in (72, 96)      # 72/96: screen defaults, not a real scan resolution
    for n, frame in enumerate(ImageSequence.Iterator(im), start=1):   # a multi-page TIFF: one page per frame
        if pages and n not in pages:
            continue
        frame = ImageOps.exif_transpose(frame.convert("RGB"))       # phone photos: turned the right way up
        img, s = _fit(np.asarray(frame))
        out.append((n, img, 72 / ((dpi if known else NOMINAL_DPI) * s), known, s > 1.02))
    return out


# ── 1. ink ─────────────────────────────────────────────────────────────────────────────────────────────────────
HUES = [  # (cv2 hue up to, colour) - coloured ink is kept apart: legends often tell symbols by colour
    (10, (1.0, 0.0, 0.0)), (22, (1.0, 0.5, 0.0)), (35, (0.8, 0.8, 0.0)), (85, (0.0, 0.6, 0.0)),
    (100, (0.0, 0.7, 0.7)), (130, (0.0, 0.0, 1.0)), (170, (0.8, 0.0, 0.8)), (181, (1.0, 0.0, 0.0)),
]
COLOURS = [(0.0, 0.0, 0.0)] + [c for _, c in HUES]


def _paper(img: np.ndarray) -> np.ndarray:
    """the paper's colour at every pixel (photos are lit unevenly, paper is off-white): a closing of a small copy,
    which the thin lines and letters vanish from"""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.resize(cv2.GaussianBlur(paper, (0, 0), 3), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def for_ocr(img: np.ndarray, paper: np.ndarray | None = None) -> np.ndarray:
    """the picture with white paper and the ink 3x darker: faint thin text (CAD stroke fonts) is found far more
    often by the text detector, and pale coloured lines show their colour clearly"""
    paper = _paper(img) if paper is None else paper
    d = cv2.GaussianBlur(np.clip(paper.astype(np.float32) - img.astype(np.float32), 0, None), (0, 0), 0.6)
    return np.ascontiguousarray(np.clip(255 - 3 * d, 0, 255).astype(np.uint8))


def read_words(img: np.ndarray, enlarged: bool) -> list[tuple[str, float, tuple]]:
    """OCR words of a picture, read several ways and the best kept: the contrast-stretched copy finds faint thin
    text, the plain picture reads small crisp text better (stretching thickens it), and a small picture that had
    to be enlarged is also read at a slightly smaller enlargement (OCR on tiny text is hit and miss)"""
    variants = [(img, 1.0), (for_ocr(img), 1.0)]
    if enlarged:
        variants.append((cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), 0.75))
    words = []
    for v, s in variants:
        for t, c, b in ocr.image_words(np.ascontiguousarray(v)):
            words.append((t, c, tuple(x / s for x in b)))
    # overlapping readings of the same place: the surest, longest one
    words.sort(key=lambda w: -(w[1] * len(w[0].strip())))
    kept: list[tuple[str, float, tuple]] = []
    for t, c, b in words:
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        if not any(max(0.0, min(b[2], k[2]) - max(b[0], k[0])) * max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
                   > 0.4 * min(area, (k[2] - k[0]) * (k[3] - k[1])) for _, _, k in kept):
            kept.append((t, c, b))
    return kept


def ink(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ink mask, colour index per pixel: 0 dark, else 1 + HUES index)"""
    paper = _paper(img)
    # ink: darker than the paper in any colour channel (a pale green line is darker in red and blue), by more than
    # the paper's own grain (scanner noise, JPEG)
    # (a slight blur first: it calms the grain far more than it fades a thin line - faint CAD lines are often
    # only 15-25 levels darker than the paper)
    diff = cv2.GaussianBlur((paper.astype(np.int16) - img.astype(np.int16)).max(axis=2).astype(np.float32), (0, 0), 0.8)
    sample = diff[::4, ::4]
    paperish = sample[sample < np.percentile(sample, 60)]
    level = float(np.median(paperish)) if paperish.size else 0.0
    grain = 1.4826 * float(np.median(np.abs(paperish - level))) if paperish.size else 3.0
    strong = diff > max(10.0, level + 6 * grain)
    # faint stretches of a line (thinner or paler there) are kept when they belong to a long stroke or touch clear
    # ink - otherwise a light circle falls apart into arcs; grain at this level only makes small specks
    weak = (diff > max(6.0, level + 3.5 * grain)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(lab[strong])] = True
    long_ = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= 15
    keep = (has_strong | long_) & (stats[:, cv2.CC_STAT_AREA] >= 4)    # (specks of dust and scanner noise out)
    keep[0] = False
    mask = keep[lab].astype(np.uint8)
    # colour from the stretched picture: a pale green line is plainly green there
    hsv = cv2.cvtColor(for_ocr(img, paper), cv2.COLOR_RGB2HSV)
    coloured = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50)
    hue_idx = np.searchsorted(np.array([h for h, _ in HUES]), hsv[:, :, 0], side="right")
    colour = np.where(coloured, 1 + np.minimum(hue_idx, len(HUES) - 1), 0).astype(np.uint8)
    return mask, colour


def wipe_words(mask: np.ndarray, words) -> np.ndarray:
    """ink pieces lying wholly inside a word of 3+ characters are its letters: wiped (short tags like "S1" or
    "AC" stay - they can be part of a symbol)"""
    boxes = [b for t, _, b in words if len(norm(t).replace(" ", "")) >= 3]
    if not boxes:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    x, y, w, h = (stats[1:, i] for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    tree = shapely.STRtree(shapely.box(*np.array([(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2) for b in boxes]).T))
    inside = tree.query(shapely.box(x, y, x + w, y + h), predicate="within")[0]
    drop = np.zeros(n, bool)
    drop[inside + 1] = True
    out = mask.copy()
    out[drop[lab]] = 0
    return out


# ── 2. thin lines -> centre lines ──────────────────────────────────────────────────────────────────────────────
# 8 neighbours as bits: N, NE, E, SE, S, SW, W, NW
_KERNEL = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)


def _neighbours(code: int) -> list[int]:
    return [(code >> i) & 1 for i in range(8)]


def _luts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step1, step2, branches, count = (np.zeros(256, np.uint8) for _ in range(4))
    for code in range(256):
        p = _neighbours(code)                      # p[0]=N p[1]=NE p[2]=E p[3]=SE p[4]=S p[5]=SW p[6]=W p[7]=NW
        b = sum(p)
        a = sum(1 for i in range(8) if p[i] == 0 and p[(i + 1) % 8] == 1)
        n, e, s, w = p[0], p[2], p[4], p[6]
        ok = 2 <= b <= 6 and a == 1
        step1[code] = ok and n * e * s == 0 and e * s * w == 0
        step2[code] = ok and n * e * w == 0 and n * s * w == 0
        branches[code] = a
        count[code] = b
    return step1, step2, branches, count


_STEP1, _STEP2, _BRANCHES, _COUNT = _luts()


def _codes(img: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, cv2.CV_32F, _KERNEL, borderType=cv2.BORDER_CONSTANT).astype(np.int32)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: the centre line of every stroke, one pixel wide"""
    # only pixels next to ones just removed can change: each pass looks at those alone, not the whole page
    sk = np.pad((mask > 0).astype(np.uint8), 1)
    ys, xs = np.nonzero(sk)
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # bit order of _KERNEL
    width = sk.shape[1]
    dys, dxs = np.array([d[0] for d in offs]), np.array([d[1] for d in offs])
    edge = _codes(sk)[ys, xs] != 255               # pixels inside a stroke can't go before its edge does
    start = ys[edge].astype(np.int64) * width + xs[edge]
    todo = [start, start.copy()]                   # pixels to look at again, for each half of a pass

    def around(flat: np.ndarray) -> np.ndarray:
        y, x = flat // width, flat % width
        f = ((y[:, None] + dys) * width + (x[:, None] + dxs)).ravel()
        return f[sk.ravel()[f] == 1]

    while len(todo[0]) or len(todo[1]):
        changed = False
        for half, lut in enumerate((_STEP1, _STEP2)):
            flat = np.unique(todo[half])
            flat = flat[sk.ravel()[flat] == 1]
            y, x = flat // width, flat % width
            code = np.zeros(len(flat), np.int32)
            for bit, (dy, dx) in enumerate(offs):
                code |= sk[y + dy, x + dx].astype(np.int32) << bit
            gone = flat[lut[code] == 1]
            sk.ravel()[gone] = 0
            # a pixel left alone keeps its verdict until something next to it goes
            near = around(gone)
            todo[half] = near
            todo[1 - half] = np.concatenate([todo[1 - half], near])
            changed |= bool(len(gone))
        if not changed:
            break
    return sk[1:-1, 1:-1]


def trace(sk: np.ndarray, min_spur: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """centre lines -> polylines (x, y in pixels) that end at line ends and where lines meet; lines meeting at a
    junction all end at the same point (the junction's centre), so they still join up. With each polyline: the
    pixels it was traced from."""
    ys, xs = np.nonzero(sk)
    n = len(ys)
    if not n:
        return []
    idx = np.full((sk.shape[0] + 2, sk.shape[1] + 2), -1, np.int32)
    idx[ys + 1, xs + 1] = np.arange(n, dtype=np.int32)
    # neighbours: N E S W first, then NE SE SW NW
    offs = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    nb = np.stack([idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in offs], axis=1)
    # a diagonal step is dropped where a pixel beside it already links the two (stair steps of a sloping line
    # would otherwise look like junctions)
    orth = nb[:, :4] >= 0
    for k, (a, b) in zip(range(4, 8), [(0, 1), (1, 2), (2, 3), (3, 0)]):
        nb[orth[:, a] | orth[:, b], k] = -1
    deg = (nb >= 0).sum(axis=1)
    node = deg != 2                                # line ends (1), junctions (3+), dots (0)
    # junction pixels next to each other are one junction: lines meeting there all end at its centre
    jmask = np.zeros(sk.shape, np.uint8)
    jmask[ys[deg >= 3], xs[deg >= 3]] = 1
    _, jlab, _, jcent = cv2.connectedComponentsWithStats(jmask, connectivity=8)
    jof = jlab[ys, xs]

    nbl = [[t for t in row if t >= 0] for row in nb.tolist()]
    node_l = node.tolist()
    seen = bytearray(n)
    used: set[tuple[int, int]] = set()
    paths: list[tuple[list[int], bool]] = []
    for s in np.flatnonzero(node).tolist():       # every branch from every end and junction, to the next one
        for t in nbl[s]:
            if node_l[t]:
                key = (s, t) if s < t else (t, s)
                if key not in used:
                    used.add(key)
                    paths.append(([s, t], False))
                continue
            if seen[t]:
                continue
            path, prev, cur = [s, t], s, t
            while not node_l[cur]:
                seen[cur] = 1
                a, b = nbl[cur]
                prev, cur = cur, (b if a == prev else a)
                path.append(cur)
            paths.append((path, False))
    for s in range(n):                             # what is left: rings (the circle of a symbol)
        if node_l[s] or seen[s]:
            continue
        path, prev, cur = [s], -1, s
        while True:
            seen[cur] = 1
            a, b = nbl[cur]
            nxt = b if a == prev else a
            if nxt == s or seen[nxt]:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        paths.append((path, True))

    out = []
    for path, closed in paths:
        p = np.array(path)
        px = np.stack([xs[p], ys[p]], axis=1)
        pts = px.astype(np.float64)
        if not closed:
            j0, j1 = jof[p[0]], jof[p[-1]]
            if j0 and j0 == j1 and len(p) <= 4:
                continue                           # inside one junction
            free = (deg[p[0]] == 1) + (deg[p[-1]] == 1)
            if free == 1 and (j0 or j1) and len(p) < min_spur:
                continue                           # a whisker the thinning left at a corner or a line end
            if j0:
                pts[0] = jcent[j0]
            if j1:
                pts[-1] = jcent[j1]
        if len(pts) < 2:
            continue
        approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 0.9, closed)[:, 0, :].astype(np.float64)
        if closed:
            approx = np.vstack([approx, approx[:1]])
        else:                                      # the junction points exactly, not rounded by the simplification
            approx[0], approx[-1] = pts[0], pts[-1]
        if len(approx) >= 2:
            out.append((approx, px))
    return out


# ── 3. thick ink -> filled shapes ──────────────────────────────────────────────────────────────────────────────
def _holeless(poly: Polygon) -> list[Polygon]:
    """a wall ring has the rooms as holes; filled shapes can't (each outline becomes solid): cut it through its
    holes into pieces without any"""
    out, todo = [], [poly]
    while todo:
        p = todo.pop()
        if p.is_empty or p.area < 1:
            continue
        if not p.interiors:
            out.append(p)
            continue
        x0, y0, x1, y1 = p.bounds
        hx = Polygon(p.interiors[0]).centroid.x
        for half in (box(x0 - 1, y0 - 1, hx, y1 + 1), box(hx, y0 - 1, x1 + 1, y1 + 1)):
            part = p.intersection(half)
            todo.extend(g for g in shapely.get_parts(part) if isinstance(g, Polygon))
    return out


def split_dots(thick: np.ndarray, max_side: int) -> np.ndarray:
    """Touching filled dots (switches stacked on a wall) print as one blob; a CAD drawing has each dot apart.
    A blob whose distance map has several round peaks, each owning about a circle's worth of the blob, is cut
    between them (wall strips have a ridge, not round peaks, and are left alone)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    out = thick.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) > max_side or area < 20:
            continue
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        d = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        r = float(d.max())
        if r < 2:
            continue
        k = max(3, int(1.4 * r) | 1)
        peaks = ((d >= cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) - 1e-3) & (d >= 0.6 * r))
        m, plab, _, cent = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
        if m - 1 < 2:
            continue
        cent = cent[1:]
        radius = d[np.clip(cent[:, 1].round().astype(int), 0, h - 1), np.clip(cent[:, 0].round().astype(int), 0, w - 1)]
        yy, xx = np.nonzero(sub)
        owner = np.argmin((xx[:, None] - cent[None, :, 0]) ** 2 + (yy[:, None] - cent[None, :, 1]) ** 2, axis=1)
        cells = np.bincount(owner, minlength=len(cent))
        if (cells > 1.9 * np.pi * np.maximum(radius, 1) ** 2).any():
            continue                               # not round dots
        cell = np.full(sub.shape, -1, np.int32)
        cell[yy, xx] = owner
        # a one-pixel cut wherever two dots' pixels meet
        cut = np.zeros(sub.shape, bool)
        cut[:, 1:] |= (cell[:, 1:] != cell[:, :-1]) & (cell[:, 1:] >= 0) & (cell[:, :-1] >= 0)
        cut[1:, :] |= (cell[1:, :] != cell[:-1, :]) & (cell[1:, :] >= 0) & (cell[:-1, :] >= 0)
        cut = cv2.dilate(cut.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0    # no corner left touching
        region = out[y:y + h, x:x + w]
        region[cut] = 0
    return out


def solids(thick: np.ndarray) -> list[list[np.ndarray]]:
    """each piece of thick ink as outlines without holes (pixels)"""
    contours, hier = cv2.findContours(thick, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:
            continue                               # a hole: handled with its outer outline
        holes = []
        k = hier[i][2]
        while k != -1:
            if len(contours[k]) >= 3:
                holes.append(contours[k][:, 0, :].astype(float))
            k = hier[k][0]
        poly = Polygon(c[:, 0, :].astype(float), holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        parts = [g for g in shapely.get_parts(poly) if isinstance(g, Polygon)]
        pieces = []
        for g in parts:
            for piece in _holeless(g.simplify(0.7)):
                pts = np.asarray(piece.exterior.coords)
                if len(pts) >= 4:
                    pieces.append(pts)
        if pieces:
            out.append(pieces)
    return out


def _line_weights(lines: list) -> list[tuple[np.ndarray, float, tuple]]:
    """A pen weight per line: widths measured from pixels wander along a line, so close widths (within about a
    third) are one weight - walls then share one weight, as in the CAD drawing (rooms are traced by weight)"""
    if not lines:
        return []
    widths = np.array([ln[1] for ln in lines])
    length = np.array([ln[3] for ln in lines], float)
    order = np.argsort(widths)
    groups: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        if widths[i] <= 1.35 * widths[groups[-1][0]]:
            groups[-1].append(i)
        else:
            groups.append([i])
    weight = np.empty(len(lines))
    for g in groups:
        g = np.array(g)
        o = np.argsort(widths[g])
        cum = np.cumsum(length[g][o])
        weight[g] = widths[g][o][np.searchsorted(cum, cum[-1] / 2)]     # the length-weighted median
    return [(ln[0], round(float(weight[i]), 2), ln[2]) for i, ln in enumerate(lines)]


# ── 4. the traced drawing as a PDF page ────────────────────────────────────────────────────────────────────────
def _mode(values: np.ndarray) -> int:
    return int(np.bincount(values).argmax()) if len(values) else 0


def vectorise(img: np.ndarray, words) -> tuple[list, list, dict]:
    """(filled shapes [(outlines, colour)], lines [(points, width px, colour)], stats) of one picture"""
    mask, colour = ink(img)
    mask = wipe_words(mask, words)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = (mask > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6)
    w = float(np.median(dist[ridge])) * 2 if ridge.any() else 2.0      # usual line width, pixels
    w = max(w, 1.0)
    k = int(round(max(2.5 * w, w + 3)))
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thin_ink = mask.copy()
    thin_ink[cv2.dilate(thick, np.ones((3, 3), np.uint8)) > 0] = 0
    # the sheet frame and border lines are often drawn heavy: thick ink running (nearly) edge to edge is not a wall
    h, wd = thick.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    frame = (stats[:, cv2.CC_STAT_WIDTH] > 0.75 * wd) | (stats[:, cv2.CC_STAT_HEIGHT] > 0.75 * h)
    frame[0] = False
    if frame.any():
        thick = thick.copy()
        thick[frame[lab]] = 0
    # the building: the big masses of solid wall (symbols are only looked for inside it - not in the title block)
    big = [i for i in range(1, n) if not frame[i] and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 0.1 * min(h, wd)]
    footprint = None
    if big:
        top = max(stats[i, cv2.CC_STAT_AREA] for i in big)
        boxes = [stats[i] for i in big if stats[i, cv2.CC_STAT_AREA] >= 0.15 * top]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        pad = int(0.02 * max(h, wd))
        footprint = (max(0, x0 - pad), max(0, y0 - pad), min(wd, x1 + pad), min(h, y1 + pad))
    # walls (big solid pieces) are left out when symbols are looked for: a solid wall "matches" a filled symbol
    walls = np.zeros_like(thick)
    big_parts = [i for i in range(1, n) if not frame[i] and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > 8 * k]
    if big_parts:
        keep = np.zeros(n, bool)
        keep[big_parts] = True
        walls = keep[lab].astype(np.uint8)
    # coloured ink is never wall: switch dots drawn against a wall face stay symbols
    walls[colour > 0] = 0
    symbol_ink = mask.copy()
    edge = max(5, k // 2 + 3) | 1                  # (and the wall's soft edges with it)
    symbol_ink[(cv2.dilate(walls, np.ones((edge, edge), np.uint8)) > 0) & (colour == 0)] = 0
    thick = split_dots(thick, max_side=int(12 * k))
    sk = thin(thin_ink)
    filled = []
    for pieces in solids(thick):
        inside = [Polygon(p).representative_point() for p in pieces]    # colour: sampled inside each piece
        xs = np.clip(np.array([q.x for q in inside]).astype(int), 0, wd - 1)
        ys = np.clip(np.array([q.y for q in inside]).astype(int), 0, h - 1)
        filled.append((pieces, COLOURS[_mode(colour[ys, xs])]))
    lines = []
    for pts, px in trace(sk, min_spur=2 * w + 2):
        width = max(1.0, float(np.median(dist[px[:, 1], px[:, 0]])) * 2)
        lines.append((pts, width, COLOURS[_mode(colour[px[:, 1], px[:, 0]])], len(px)))
    lines = _line_weights(lines)
    return filled, lines, {"line_px": w, "solid_px": k, "mask": mask, "symbol_ink": symbol_ink, "colour": colour,
                           "footprint": footprint, "walls": walls}


# ── 5. symbols: each legend symbol, cut from the picture, looked for over the plan ─────────────────────────────
# chamfer matching (pixels at the working size, symbol ~32 px): how far the symbol's strokes are from the plan's,
# on average, and the plan's strokes there from the symbol's
FORWARD_MAX = 2.0
BACKWARD_MAX = 3.0
COVER_MIN = 0.75               # share of the symbol's ink found on the plan (with a pixel of slack)
SYMBOL_PX = 20.0               # symbols are found at about this size ...
FINE_PX = 48.0                 # ... and told apart at this size
MATCH_MIN = 0.38               # score (1 / (1 + cost)) of a match (measured on a scanned test plan)
COARSE_MIN = 0.30              # the quick search's score worth a closer look
MAX_PLACES = 2000              # places looked at closely per page (strongest first)
CLIP = 6.0


def _edges(b: np.ndarray) -> np.ndarray:
    """the outline of ink (a thin line is its own outline; a filled dot becomes a ring)"""
    return (b - cv2.erode(b, np.ones((3, 3), np.uint8))).astype(np.uint8)


def templates(mask: np.ndarray, colour: np.ndarray, rows, pt: float) -> list[dict]:
    """the symbol(s) drawn in each legend row, as small ink pictures: {kind, desc, ink, colour, size}"""
    out = []
    H, W = mask.shape
    for kind, desc, strip in rows:
        x0, y0 = max(0, int(strip.x0 / pt)), max(0, int(strip.y0 / pt))
        x1, y1 = min(W, int(strip.x1 / pt)), min(H, int(strip.y1 / pt))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        cell = mask[y0:y1, x0:x1].copy()
        h, w = cell.shape
        # table borders: straight lines running (nearly) the whole cell - a scan is often a little tilted, so the
        # line is thickened across first (a tilted line steps by a pixel or two over the cell)
        hor = cv2.morphologyEx(cv2.dilate(cell, np.ones((5, 1), np.uint8)), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.7 * w)), 1)))
        ver = cv2.morphologyEx(cv2.dilate(cell, np.ones((1, 5), np.uint8)), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(0.8 * h)))))
        cell[cv2.dilate(hor | ver, np.ones((3, 3), np.uint8)) > 0] = 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cell, connectivity=8)
        biggest = max([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0)
        comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(6, 0.03 * biggest)]
        if not comps:
            continue
        # a row can show several symbols side by side (normal / smart / dimmer switch): split at wide gaps
        comps.sort(key=lambda c: c[0])
        groups, gap = [[comps[0]]], 0.25 * h
        for c in comps[1:]:
            right = max(g[0] + g[2] for g in groups[-1])
            if c[0] - right < gap:
                groups[-1].append(c)
            else:
                groups.append([c])
        boxes = []
        for g in groups:
            gx0 = min(c[0] for c in g); gy0 = min(c[1] for c in g)
            gx1 = max(c[0] + c[2] for c in g); gy1 = max(c[1] + c[3] for c in g)
            boxes.extend(_split_repeats(cell, (gx0, gy0, gx1, gy1)))
        sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
        top = max(sizes) if sizes else 0
        for (gx0, gy0, gx1, gy1), size in zip(boxes, sizes):
            if size < 0.35 * top or size < 0.15 * h:
                continue                           # a speck or a tick, not a symbol
            if abs((gy0 + gy1) / 2 - h / 2) > 0.3 * h or gy0 <= 0 or gy1 >= h:
                continue                           # not on the row's line, or cut by the cell (the next row's)
            if min(gx1 - gx0, gy1 - gy0) < 0.25 * size:
                continue                           # a sliver (a piece of a border line)
            ink = cell[max(0, gy0 - 2):gy1 + 2, max(0, gx0 - 2):gx1 + 2]
            cols = colour[y0:y1, x0:x1][max(0, gy0 - 2):gy1 + 2, max(0, gx0 - 2):gx1 + 2][ink > 0]
            out.append({"kind": kind, "desc": desc, "ink": ink, "colour": _mode(cols), "size": size,
                        "at": ((x0 + gx0) * pt, (y0 + gy0) * pt)})
    return out


def _split_repeats(cell: np.ndarray, b: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """A row showing the same symbol in a few variants side by side (a dot with differently drawn ticks: normal /
    smart / dimmer switch) is cut halfway between its dots, one symbol each"""
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    if w < 1.5 * h:
        return [b]
    part = cell[y0:y1, x0:x1]
    k = max(3, int(0.15 * h) | 1)
    dots = cv2.morphologyEx(part, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(dots, connectivity=8)
    if n - 1 < 2:
        return [b]
    areas = stats[1:, cv2.CC_STAT_AREA]
    same = areas >= 0.5 * np.median(areas)
    xs = np.sort(cent[1:, 0][same])
    if len(xs) < 2:
        return [b]
    cuts = [0] + [int((a + c) / 2) for a, c in zip(xs[:-1], xs[1:])] + [w]
    out = []
    for a, c in zip(cuts[:-1], cuts[1:]):
        cols = np.flatnonzero(part[:, a:c].any(axis=0))
        rows = np.flatnonzero(part[:, a:c].any(axis=1))
        if len(cols) and len(rows):
            out.append((x0 + a + cols[0], y0 + rows[0], x0 + a + cols[-1] + 1, y0 + rows[-1] + 1))
    return out or [b]


class _Plan:
    """the plan's ink at one working size, ready for chamfer matching"""

    def __init__(self, plan: np.ndarray, f: float, symbol_px: float):
        self.f = f
        small = (cv2.resize(plan.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8)
        self.shape = small.shape
        edges = _edges(small)
        self.reach = np.minimum(cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3), CLIP).astype(np.float32)
        # long straight lines (walls, wires, a tile grid) don't count as extra strokes in a symbol's window
        n = int(1.6 * symbol_px)
        long_ = (cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1)))
                 | cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n))))
        self.loose = (edges & (cv2.dilate(long_, np.ones((3, 3), np.uint8)) == 0)).astype(np.float32)
        self.edges = edges.astype(np.float32)
        self.near = cv2.dilate(small, np.ones((3, 3), np.uint8)).astype(np.float32)    # (a pixel of slack)
        self.loose_sum = cv2.integral(self.loose, sdepth=cv2.CV_64F)
        self.edges_sum = cv2.integral(self.edges, sdepth=cv2.CV_64F)


def _window_sums(ii: np.ndarray, sl, h: int, w: int, shape) -> np.ndarray:
    """the sum over every h x w window (top-left corners, like matchTemplate) from an integral image"""
    y0 = sl[0].start or 0
    x0 = sl[1].start or 0
    rows, cols = shape
    a = ii[y0:y0 + rows, x0:x0 + cols]
    b = ii[y0:y0 + rows, x0 + w:x0 + w + cols]
    c = ii[y0 + h:y0 + h + rows, x0:x0 + cols]
    d = ii[y0 + h:y0 + h + rows, x0 + w:x0 + w + cols]
    return (d - b - c + a).astype(np.float32)


def _shapes(t: dict, f: float) -> list[dict]:
    """a legend symbol at working size f, in its four quarter turns"""
    tb = (cv2.resize(t["ink"].astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8)
    # a margin around the symbol: strokes just outside it count against it (a small symbol sitting inside a bigger
    # one - a switch's dot and tick inside a light's circle - is not that small symbol)
    m = max(3, int(0.25 * max(tb.shape)))
    tb = np.pad(tb, m)
    te = _edges(tb)
    if te.sum() < 8:
        return []
    reach = np.minimum(cv2.distanceTransform(1 - te, cv2.DIST_L2, 3), CLIP).astype(np.float32)
    out = []
    for k in range(4):
        r = np.rot90(te, k)
        if any(o["e"].shape == r.shape and np.logical_and(o["e"] > 0, r > 0).sum() > 0.9 * te.sum() for o in out):
            out.append(None)                       # looks the same as a turn already listed (a round symbol)
            continue
        out.append({"e": np.ascontiguousarray(r).astype(np.float32),
                    "reach": np.ascontiguousarray(np.rot90(reach, k)),
                    "ink": np.ascontiguousarray(np.rot90(tb, k)).astype(np.float32),
                    "own": float(te.sum()), "solid": float(tb.sum()), "m": m})
    return out


def _cost(P: _Plan, s: dict, box=None) -> tuple[np.ndarray, np.ndarray]:
    """(cost, acceptable) of the symbol s at every position of the plan (or of box = (x0, y0, x1, y1) in it)"""
    sl = (slice(None), slice(None)) if box is None else (slice(box[1], box[3]), slice(box[0], box[2]))
    e, own = s["e"], s["own"]
    h, w = e.shape
    # every stroke of the symbol close to a stroke of the plan ...
    forward = cv2.matchTemplate(P.reach[sl], e, cv2.TM_CCORR) / own
    # ... the plan's strokes there close to the symbol's (a wire or wall through it costs nothing) ...
    count = _window_sums(P.loose_sum, sl, h, w, forward.shape)
    backward = cv2.matchTemplate(P.loose[sl], s["reach"], cv2.TM_CCORR) / np.maximum(count, 1)
    strokes = _window_sums(P.edges_sum, sl, h, w, forward.shape)
    # ... and the symbol's own ink really there (a filled dot needs filled ink, not just a ring of strokes)
    cover = cv2.matchTemplate(P.near[sl], s["ink"], cv2.TM_CCORR) / max(s["solid"], 1)
    cost = forward + 0.5 * backward + 2 * (1 - np.minimum(cover, 1))
    ok = (forward <= FORWARD_MAX) & (backward <= BACKWARD_MAX) & (strokes >= 0.6 * own) & (cover >= COVER_MIN)
    return cost, ok


def match(mask: np.ndarray, colour: np.ndarray, temps: list[dict], region, skip) -> list[tuple[str, float, float, float]]:
    """(kind, x, y, score) of every place a legend symbol is drawn (pixels), at any quarter turn; each place once.
    region: (x0, y0, x1, y1) to look in (the building); skip: pixel boxes to leave out (the legend).
    A quick search at a small size finds the places; each is then judged again in full detail against every
    legend symbol (at the small size, fine details such as a socket's three bars run together)."""
    if not temps:
        return []
    plan = np.zeros(mask.shape, np.uint8)
    x0, y0, x1, y1 = region
    plan[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    for a, b, c, d in skip:
        plan[max(0, b):d, max(0, a):c] = 0
    typical = float(np.median([t["size"] for t in temps]))
    # only the part of the page with ink in it (plus a margin) is searched
    ys, xs = np.nonzero(plan[::4, ::4])
    if not len(ys):
        return []
    pad = int(2 * typical)
    ox, oy = max(0, xs.min() * 4 - pad), max(0, ys.min() * 4 - pad)
    plan = plan[oy:min(plan.shape[0], ys.max() * 4 + 4 + pad), ox:min(plan.shape[1], xs.max() * 4 + 4 + pad)]
    # 1. the places: a quick search at a small size
    coarse = _Plan(plan, float(np.clip(SYMBOL_PX / typical, 0.15, 1.0)), SYMBOL_PX)
    cands = []                                     # (score, x, y, size, symbol, turn)
    for ti, t in enumerate(temps):
        for k, s in enumerate(_shapes(t, coarse.f)):
            if s is None:
                continue
            e = s["e"]
            if e.shape[0] >= coarse.shape[0] or e.shape[1] >= coarse.shape[1]:
                continue
            cost, ok = _cost(coarse, s)
            win = np.ones((max(3, e.shape[0] // 2) | 1, max(3, e.shape[1] // 2) | 1), np.uint8)
            best = (cost <= cv2.erode(cost, win) + 1e-6) & ok
            best &= cost <= 1.0 / COARSE_MIN - 1.0
            for yy, xx in zip(*np.nonzero(best)):
                cands.append((1.0 / (1.0 + float(cost[yy, xx])), (xx + e.shape[1] / 2) / coarse.f,
                              (yy + e.shape[0] / 2) / coarse.f, (e.shape[1] - 2 * s["m"]) / coarse.f, ti, k))
    places = _one_each([c[:4] for c in cands], typical, keep_index=True)[:MAX_PLACES]
    # 2. each place judged again in detail - against the symbols (and turns) that came close there
    fine = _Plan(plan, float(np.clip(FINE_PX / typical, 0.15, 1.0)), FINE_PX)
    shapes = [_shapes(t, fine.f) for t in temps]
    grid: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(cands):
        grid.setdefault((int(c[1] // typical), int(c[2] // typical)), []).append(i)
    found = []
    for pi in places:
        _, cx, cy, size = cands[pi][:4]
        fx, fy = cx * fine.f, cy * fine.f
        close = sorted((j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        for j in grid.get((int(cx // typical) + dx, int(cy // typical) + dy), [])
                        if (cands[j][1] - cx) ** 2 + (cands[j][2] - cy) ** 2 < (0.55 * max(size, cands[j][3])) ** 2),
                       key=lambda j: -cands[j][0])
        tried = list(dict.fromkeys((cands[j][4], cands[j][5]) for j in close))[:4]   # the closest few
        best = None
        for ti, k in tried:
            if k >= len(shapes[ti]) or shapes[ti][k] is None:
                continue
            s = shapes[ti][k]
            h, w = s["e"].shape
            slack = max(3, int(0.15 * max(h, w)))
            bx0, by0 = int(fx - w / 2) - slack, int(fy - h / 2) - slack
            bx1, by1 = bx0 + w + 2 * slack, by0 + h + 2 * slack
            if bx0 < 0 or by0 < 0 or bx1 > fine.shape[1] or by1 > fine.shape[0]:
                continue
            cost, ok = _cost(fine, s, (bx0, by0, bx1, by1))
            cost = np.where(ok, cost, np.inf)
            yy, xx = np.unravel_index(int(np.argmin(cost)), cost.shape)
            c = float(cost[yy, xx])
            if np.isfinite(c) and (best is None or c < best[0]):
                best = (c, ti, (bx0 + xx + w / 2) / fine.f, (by0 + yy + h / 2) / fine.f, (w - 2 * s["m"]) / fine.f)
        if best is None:
            continue
        c, ti, bx, by, bsize = best
        score = 1.0 / (1.0 + c)
        if score < MATCH_MIN:
            continue
        t = temps[ti]
        if t["colour"]:                            # a coloured legend symbol: the match must not be another colour
            hh = int(bsize / 2)
            gx, gy = int(bx + ox), int(by + oy)    # (in the whole picture)
            win = colour[max(0, gy - hh):gy + hh, max(0, gx - hh):gx + hh]
            inks = win[mask[max(0, gy - hh):gy + hh, max(0, gx - hh):gx + hh] > 0]
            # (its own colour among the coloured ink there, and not plain black / grey ink)
            coloured = inks[inks > 0]
            if not len(coloured) or _mode(coloured) != t["colour"] or len(coloured) < 0.3 * len(inks):
                continue
        found.append((score, bx, by, bsize, t["kind"]))
    kept = _one_each([(sc, x, y, sz) for sc, x, y, sz, _ in found], typical, keep_index=True)
    return [(found[i][4], found[i][1] + ox, found[i][2] + oy, found[i][0]) for i in kept]


def _one_each(places, typical: float, keep_index: bool = False):
    """strongest first; a weaker place overlapping a kept one is the same symbol (or a part of it)"""
    order = sorted(range(len(places)), key=lambda i: -places[i][0])
    kept: list[int] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell = max(4.0, typical)
    for i in order:
        score, cx, cy, size = places[i]
        gx, gy = int(cx // cell), int(cy // cell)
        clash = any((places[j][1] - cx) ** 2 + (places[j][2] - cy) ** 2 < (0.55 * max(size, places[j][3])) ** 2
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((gx + dx, gy + dy), []))
        if clash:
            continue
        kept.append(i)
        grid.setdefault((gx, gy), []).append(i)
    return kept if keep_index else [places[i] for i in kept]


def _absurd_size(text: str) -> bool:
    """a room size misread by OCR ("I89 X 14-5" for 18'-9" x 14'-6"): a side over 60 ft"""
    d = parse_dimensions(text)
    if not d:
        return False
    nums = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", text)]
    return d[1] > 3600 or any(v > 60 for v in nums[:1] + nums[-2:-1])


SINGLE_FEET = re.compile(r"^(\d{2,3})\s*(?:'|’|′|FT)\s*(?:-?\s*(\d{1,2})\s*(?:\"|''|”|″)?)?$")
TYPICAL_ROOM_SQFT = 150.0


def picture_scale(lines_pt, footprint, rooms_px: list[float], pt: float, views_pt: list | None = None) -> tuple[int | None, str]:
    """(1:N, how it was worked out) for a picture whose print size is unknown, or (None, "") to leave it to the
    room dimensions written on the plan. An overall dimension written on the plan ("50'", "90'") is taken as the
    building's length; else a typical room (150 sq ft) sets the size."""
    values = []                                    # (feet, written upright = along the building's height, where)
    for t, r in lines_pt:
        if m := SINGLE_FEET.match(t.text.strip().upper()):
            values.append((int(m.group(1)) + int(m.group(2) or 0) / 12, r.height > r.width, ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)))
    values = [v for v in values if 15 <= v[0] <= 400]
    if values and views_pt:
        # several drawings on the sheet: the dimension spans the drawing it is written in (drawn to the same scale)
        for feet, upright, (cx, cy) in sorted(values, reverse=True):
            # (inside it, or written just outside its edge - a dimension line above or beside the drawing)
            v = next((b for b in views_pt if b[0] - 0.08 * (b[2] - b[0]) <= cx <= b[2] + 0.08 * (b[2] - b[0])
                      and b[1] - 0.08 * (b[3] - b[1]) <= cy <= b[3] + 0.08 * (b[3] - b[1])), None)
            if v:
                side = 0.85 * ((v[3] - v[1]) if upright else (v[2] - v[0]))      # (less its margin and texts)
                n = round(feet * 864 / max(side, 1))
                if 20 <= n <= 2000:
                    return n, f"from the dimension {feet:g}' written on the sheet"
    if values and footprint and not views_pt:
        x0, y0, x1, y1 = footprint
        pad = 0.02 * max(x1 - x0, y1 - y0) / 0.96
        feet, upright, _ = max(values)
        side_pt = ((y1 - y0) if upright else (x1 - x0)) - 2 * pad
        n = round(feet * 864 / max(side_pt * pt, 1))
        if 20 <= n <= 2000:
            return n, f"from the overall dimension {feet:g}' written on the plan"
    dims = sum(1 for t, _ in lines_pt if parse_dimensions(t.text))
    if dims >= 2 or not rooms_px:
        return None, ""
    ft_pt = (float(np.median(rooms_px)) * pt * pt / TYPICAL_ROOM_SQFT) ** 0.5
    n = round(864 / max(ft_pt, 1e-6))
    return (n, "estimated from the number and size of the rooms (no sizes could be read)") if 20 <= n <= 2000 else (None, "")


def unnamed_rooms(walls: np.ndarray, footprint, lines_pt, pt: float, mask: np.ndarray | None = None,
                  line_px: float = 0.0, rooms_px: list | None = None, outlines: list | None = None) -> list:
    """_unnamed_rooms from the walls; when they enclose (almost) nothing - walls drawn as thin double lines - from
    all the drawing's lines"""
    rooms_px = rooms_px if rooms_px is not None else []
    outlines = outlines if outlines is not None else []
    found = _unnamed_rooms(walls, footprint, lines_pt, pt, mask, line_px, rooms_px, outlines)
    if len(rooms_px) < 2 and mask is not None and footprint is not None:
        rooms_px.clear(); outlines.clear()
        found = _unnamed_rooms(mask, footprint, lines_pt, pt, None, 0.0, rooms_px, outlines)
    return found


def _unnamed_rooms(walls: np.ndarray, footprint, lines_pt, pt: float, mask: np.ndarray | None = None,
                   line_px: float = 0.0, rooms_px: list | None = None, outlines: list | None = None) -> list:
    """Spaces enclosed by solid walls that have no room name read in them (the names too small or blurred to
    read): a text line "ROOM n" at a point inside each, so they are traced and counted as rooms all the same.
    rooms_px (if given) gets the area in pixels of every room space found, named or not."""
    rooms_px = rooms_px if rooms_px is not None else []
    outlines = outlines if outlines is not None else []      # every room space's outline (PDF points)
    if footprint is None:
        return []
    x0, y0, x1, y1 = footprint
    w = (walls[y0:y1, x0:x1] > 0).astype(np.uint8)
    if line_px:
        # a blurry, enlarged picture makes every line look thick, so the solid-wall test can miss the walls: here
        # walls are ink clearly heavier than the usual line that runs long (furniture and text are short)
        heavy = cv2.morphologyEx(mask[y0:y1, x0:x1], cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(1.6 * line_px)),) * 2))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(heavy, connectivity=8)
        longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        keep = longest >= 0.06 * min(heavy.shape)
        keep[0] = False
        w = w | keep[lab].astype(np.uint8)
    if w.sum() < 100:
        return []
    # door openings closed: about four wall thicknesses (a 9" wall, a 3' door)
    dist = cv2.distanceTransform(w, cv2.DIST_L2, 3)
    thick = 2 * float(np.median(dist[(w > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6)]))
    if line_px:                                    # (furniture joined to a wall makes it look thicker)
        thick = min(thick, 2 * line_px)
    k = max(5, int(4.5 * max(thick, 2))) | 1
    closed = cv2.morphologyEx(w, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    free = (closed == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    total = free.size
    named = [(((r.x0 + r.x1) / 2) / pt - x0, ((r.y0 + r.y1) / 2) / pt - y0) for t, r in lines_pt if room_type(t.text)]
    out = []
    rooms_px.clear()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= free.shape[1] or y + bh >= free.shape[0]:
            continue                               # open to the outside of the building
        if not 0.004 * total <= area <= 0.4 * total or min(bw, bh) < 0.8 * k:
            continue                               # a duct or a gap, or not a room
        rooms_px.append(float(area))
        cell = (lab[y:y + bh, x:x + bw] == i)
        cs, _ = cv2.findContours(cell.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2.0, True)[:, 0, :]
            if len(c) >= 3:
                outlines.append([((x0 + x + px_) * pt, (y0 + y + py_) * pt) for px_, py_ in c])
        if any(0 <= int(py) - y < bh and 0 <= int(px) - x < bw and cell[int(py) - y, int(px) - x] for px, py in named):
            continue                               # its name was read
        d = cv2.distanceTransform(np.pad(cell.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)    # the point deepest inside it
        out.append(((x0 + x + cx) * pt, (y0 + y + cy) * pt))
    lines = []
    for j, (px, py) in enumerate(sorted(out, key=lambda p: (round(p[1] / 50), p[0])), start=1):
        h = 4.0
        lines.append((Text(f"ROOM {j}", px, py, h), pymupdf.Rect(px - 2 * h, py - h / 2, px + 2 * h, py + h / 2)))
    return lines


def _joined(polylines: list[np.ndarray]) -> list[int]:
    """for each polyline, a label shared by all polylines connected to it end to end (lines meeting at a junction
    end at exactly the same point)"""
    parent = list(range(len(polylines)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    at: dict[tuple[int, int], int] = {}
    for i, p in enumerate(polylines):
        for x, y in (p[0], p[-1]):
            key = (int(round(x * 2)), int(round(y * 2)))
            j = at.setdefault(key, i)
            if j != i:
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(polylines))]


def to_pdf(pages: list[tuple[np.ndarray, float, list, list]]) -> bytes:
    """the traced drawings, one page each: (image, points per pixel, filled shapes, lines)"""
    # the page's drawing commands are written directly (PDF y runs upwards): far quicker than one call per line
    doc = pymupdf.open()
    for img, pt, filled, lines in pages:
        height = img.shape[0] * pt
        page = doc.new_page(width=img.shape[1] * pt, height=height)

        def path(p: np.ndarray) -> str:
            xy = np.column_stack([p[:, 0] * pt, height - p[:, 1] * pt])
            cmds = [f"{xy[0, 0]:.2f} {xy[0, 1]:.2f} m"] + [f"{x:.2f} {y:.2f} l" for x, y in xy[1:]]
            return " ".join(cmds)

        ops = ["q 1 J 1 j"]
        for pieces, col in filled:            # one path per piece of thick ink (a wall network, a dot)
            ops.append("{:.2f} {:.2f} {:.2f} rg ".format(*col) + " ".join(path(p) + " h" for p in pieces) + " f")
        # lines that meet (and share weight and colour) are one path, like a wall drawn in one go in CAD: rooms
        # are traced from long paths only, and the trace cuts a wall wherever something touches it
        groups: dict[tuple, list[np.ndarray]] = {}
        for i, root in enumerate(_joined([pts for pts, _, _ in lines])):
            pts, width, col = lines[i]
            groups.setdefault((root, width, col), []).append(pts)
        for (_, width, col), parts in groups.items():
            ops.append("{:.2f} {:.2f} {:.2f} RG {:.3f} w ".format(*col, width * pt) + " ".join(path(p) for p in parts) + " S")
        ops.append("Q")
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, "\n".join(ops).encode())
        doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    return doc.tobytes(garbage=0, deflate=True)


def read(data: bytes, kind: str, mapper: SymbolMapper, log: list[str], warnings: list[str],
         pages: list[int] | None = None) -> tuple[dict, list[Sheet], list[dict], bytes]:
    """kind: "scan" (a PDF of scanned pages) or "image". Returns pdf_reader's result plus the traced PDF."""
    pics = pictures(data, kind, pages)
    what = "Scanned PDF" if kind == "scan" else "Picture"
    log.append(f"{what}, {len(pics)} page(s) - read from the image: words by OCR, lines and walls traced from the ink")
    traced, ocr_lines, points_of, outlines_of = [], {}, {}, {}
    unknown_size = False
    small_pages, unreadable, unnamed_pages, scale_notes = [], [], [], []
    for i, (page_no, img, pt, known, enlarged) in enumerate(pics, start=1):
        if ocr.progress:
            ocr.progress(i, len(pics))
        words = read_words(img, enlarged)
        if enlarged:
            small_pages.append(page_no)
        if sum(1 for t, _, _ in words if len(t.strip()) >= 4) < 5:
            unreadable.append(page_no)
        filled, lines, st = vectorise(img, words)
        lines_pt = ocr.to_lines(words, pt)
        lines_pt = [(t, r) for t, r in lines_pt if not _absurd_size(t.text)]
        if not known:      # the paper size is unknown, so a written "SCALE 1:100" can't give sizes
            lines_pt = [(t, r) for t, r in lines_pt if "SCALE" not in norm(t.text)]
            unknown_size = True
        n = len(traced) + 1
        ocr_lines[n] = lines_pt
        traced.append((img, pt, filled, lines))
        # symbols: the legend rows' symbols, matched as pictures over the building
        rows = pdf_reader.legend_rows(lines_pt, mapper)
        temps = templates(st["mask"], st["colour"], rows, pt)
        found = []
        if temps:
            H, W = st["mask"].shape
            region = st["footprint"] or (0, 0, W, H)
            legend_px = []
            for _, _, strip in rows:              # the legend table itself (and a margin) is not the plan
                legend_px.append((int(strip.x0 / pt) - 10, int(strip.y0 / pt) - 10, int(strip.x1 / pt) + 10, int(strip.y1 / pt) + 10))
            if legend_px:
                xs0 = min(b[0] for b in legend_px); ys0 = min(b[1] for b in legend_px)
                xs1 = max(b[2] for b in legend_px); ys1 = max(b[3] for b in legend_px)
                legend_px = [(xs0, ys0, xs1 + int(0.4 * (xs1 - xs0)), ys1)]   # (with the description column)
            found = match(st["symbol_ink"], st["colour"], temps, region, legend_px)
            points_of[n] = [PointItem(kind, x * pt, y * pt) for kind, x, y, _ in found]
            # short texts inside a matched symbol ("TV" in its box, "AC") are part of it: not room names, and not
            # counted again as text symbols
            reach = (0.6 * float(np.median([t["size"] for t in temps]))) ** 2
            ocr_lines[n] = [(t, r) for t, r in lines_pt if len(t.text.strip()) > 4 or not any(
                ((r.x0 + r.x1) / 2 / pt - x) ** 2 + ((r.y0 + r.y1) / 2 / pt - y) ** 2 < reach for _, x, y, _ in found)]
        rooms_px: list[float] = []
        outlines_of[n] = []
        # several drawings on the sheet (floor plans side by side, elevations, an area plan): rooms are looked for
        # in each floor plan on its own - not across the sheet, and not in elevations or sections
        regions = [st["footprint"]]
        page_items = geo.strokes(pymupdf.open(stream=to_pdf([(img, pt, filled, lines)]), filetype="pdf")[0].get_drawings())
        views = pdf_reader._views(page_items, ocr_lines[n], pymupdf.Rect(0, 0, img.shape[1] * pt, img.shape[0] * pt), None)
        if len(views) >= 2 and sum(v["titled"] for v in views) >= 2:
            regions = [tuple(int(c / pt) for c in v["bbox"]) for v in views
                       if v["kind"] not in pdf_reader.NOT_PLAN and v["kind"] != "area"]
        extra = []
        for region in regions:
            sizes, shapes_here = [], []
            extra += unnamed_rooms(st["walls"], region, ocr_lines[n], pt, st["mask"], st["line_px"], sizes, shapes_here)
            rooms_px += sizes
            outlines_of[n] += shapes_here
        if len(regions) > 1:                       # numbered across the sheet (each plan is its own floor anyway)
            extra = [(Text(f"ROOM {j}", t.x, t.y, t.h), r) for j, (t, r) in enumerate(extra, start=1)]
        if extra:
            ocr_lines[n] = ocr_lines[n] + extra
            unnamed_pages.append(page_no)
        # the drawing's scale, when the picture can't give it (no print size, or no "SCALE 1:100" read)
        if not known or not any("SCALE" in norm(t.text) and ":" in t.text for t, _ in ocr_lines[n]):
            scale, how = picture_scale(ocr_lines[n], st["footprint"], rooms_px, pt,
                                       [v["bbox"] for v in views] if len(regions) > 1 else None)
            if scale:
                ocr_lines[n] = [(t, r) for t, r in ocr_lines[n] if "SCALE" not in norm(t.text)]
                ocr_lines[n].append((Text(f"SCALE 1:{scale}", 1, 1, 1), pymupdf.Rect(0, 0, 2, 2)))
                scale_notes.append(f"Page {page_no}: room sizes {how} - check them.")
                unknown_size = False
        counts = Counter(k for k, *_ in found)
        log.append(f"Page {page_no}: {len(words)} word(s) read, {len(lines)} line(s) and {len(filled)} solid shape(s) "
                   f"traced (lines about {st['line_px']:.1f} px wide); {len(temps)} legend symbol(s) cut out and matched: "
                   + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none found"))
    pdf = to_pdf(traced)
    project, sheets, legend = pdf_reader.read(pdf, mapper, log, warnings, None, ocr_lines, points_of, outlines_of)
    if scale_notes:
        project.pop("scale", None)                 # (our own estimate, not written on the drawing)
    warnings.insert(0, f"This drawing was read from a {'scan' if kind == 'scan' else 'picture'}: words by OCR and "
                       "lines traced from the image - check room names and counts (a DXF or CAD PDF reads best).")
    if unknown_size:
        warnings.insert(1, "The picture doesn't say its print size, so room areas come from the dimensions written "
                           "in the rooms (or an estimate) - check the areas.")
    pages_txt = pdf_reader._pages
    if unreadable:
        warnings.insert(0, f"Page(s) {pages_txt(unreadable)}: almost no words could be read - the picture is too small "
                           "or blurry. Send the original file (DXF / PDF from CAD) or a sharp scan of at least "
                           "2000 pixels across.")
    for note in reversed(scale_notes):
        warnings.insert(0, note)
    if unnamed_pages:
        warnings.insert(0, f"Page(s) {pages_txt(unnamed_pages)}: some rooms' names could not be read - they are listed "
                           "as ROOM 1, ROOM 2 ...; rename them and check their sizes.")
    if unreadable:
        pass
    elif small_pages:
        warnings.insert(0, f"Page(s) {pages_txt(small_pages)}: the picture is small, so small text may be misread. "
                           "WhatsApp shrinks photos - send the drawing as a Document, or the original file.")
    return project, sheets, legend, pdf
