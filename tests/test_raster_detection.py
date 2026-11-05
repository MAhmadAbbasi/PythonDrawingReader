"""Pictures and scans of a drawing (app/extract/raster_reader.py): the CAD-style test plan (tests/cad_fixture.py),
printed and "scanned" - 300 dpi, a little tilted, blurred and grainy - must read nearly like the CAD PDF itself."""
from __future__ import annotations

import io

import numpy as np
import pymupdf
import pytest

from app.extract import extract, ocr
from app.extract import raster_reader as rr
from tests import cad_fixture

pytestmark = pytest.mark.skipif(not ocr.available(), reason="needs the OCR reader (rapidocr-onnxruntime)")


def _scan(pdf: bytes, dpi: int = 300, angle: float = 0.5) -> np.ndarray:
    import cv2
    pix = pymupdf.open(stream=pdf, filetype="pdf")[0].get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h), borderValue=(255, 255, 255))
    img = cv2.GaussianBlur(img, (0, 0), 0.6).astype(np.float32) * np.array([0.96, 0.96, 0.93])
    noise = np.random.default_rng(3).normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    from PIL import Image
    pdf = cad_fixture.make_cad_pdf(tmp_path_factory.mktemp("scan") / "plan.pdf").read_bytes()
    buf = io.BytesIO()
    Image.fromarray(_scan(pdf)).save(buf, format="JPEG", quality=88, dpi=(300, 300))
    return buf.getvalue()


def test_scanned_plan_symbols(scanned):
    ex = extract("plan.jpg", scanned)
    assert ex["fileType"] == "image"
    got = ex["totals"]["items"]
    expected: dict[str, int] = {}
    for room in cad_fixture.EXPECTED.values():
        for k, v in room.items():
            expected[k] = expected.get(k, 0) + v
    for k, v in cad_fixture.UNASSIGNED.items():
        expected[k] = expected.get(k, 0) + v
    # symbols drawn clearly apart read exactly
    for kind in ("light", "fan", "exhaust_fan", "tv_point", "ac_point", "curtain"):
        assert got.get(kind, 0) == expected[kind], (kind, got, expected)
    # sockets and switch boards sit on the wall lines: nearly all found, and none invented
    sockets = got.get("socket_5A", 0) + got.get("socket_15A", 0)
    assert expected["socket_5A"] + expected["socket_15A"] - 3 <= sockets <= expected["socket_5A"] + expected["socket_15A"]
    assert expected["switch"] - 3 <= got.get("switch", 0) <= expected["switch"]
    assert any("read from a picture" in w for w in ex["warnings"])


def test_scanned_plan_rooms(scanned):
    ex = extract("plan.jpg", scanned)
    areas = {a["name"]: a for f in ex["floors"] for a in f["areas"]}
    for room in ("MASTER BED ROOM", "LOUNGE", "KITCHEN", "BATH", "PASSAGE", "GUEST ROOM", "DINING"):
        assert room in areas, sorted(areas)
    assert not any(name.startswith("TV") for name in areas)     # the letters in a symbol are not a room
    # rooms traced from the walls get their symbols
    assert areas["LOUNGE"]["items"].get("light") == cad_fixture.EXPECTED["LOUNGE"]["light"]
    assert areas["DINING"]["items"].get("light") == cad_fixture.EXPECTED["DINING"]["light"]
    assert areas["DINING"]["areaSqft"] == pytest.approx(154, abs=2)        # from "14'x11'" written in it


def test_scanned_pdf_reads_like_the_picture(scanned):
    doc = pymupdf.open()
    page = doc.new_page(width=1191, height=842)
    page.insert_image(page.rect, stream=scanned)
    ex = extract("scan.pdf", doc.tobytes())
    assert ex["fileType"] == "scan"
    assert ex["totals"]["items"].get("light") == 25


def test_touching_dots_are_split():
    import cv2
    img = np.zeros((80, 160), np.uint8)
    for x in (30, 54, 78, 102):         # four dots in a row, just touching (switches stacked on a wall)
        cv2.circle(img, (x, 40), 12, 1, -1)
    out = rr.split_dots(img, max_side=200)
    n, *_ = cv2.connectedComponentsWithStats(out, connectivity=8)
    assert n - 1 == 4
    wall = np.zeros((80, 300), np.uint8)
    wall[30:50, 10:290] = 1             # a wall strip is not cut into "dots"
    n, *_ = cv2.connectedComponentsWithStats(rr.split_dots(wall, max_side=400), connectivity=8)
    assert n - 1 == 1


def test_thinning_matches_whole_image_method():
    import cv2
    img = np.zeros((120, 200), np.uint8)
    cv2.circle(img, (60, 60), 30, 1, 5)
    cv2.line(img, (100, 20), (190, 100), 1, 4)
    sk = rr.thin(img)
    ref = img.copy()
    while True:                         # plain Zhang-Suen over the whole image
        changed = False
        for lut in (rr._STEP1, rr._STEP2):
            r = (ref == 1) & (lut[rr._codes(ref)] == 1)
            if r.any():
                ref[r] = 0
                changed = True
        if not changed:
            break
    assert (sk == ref).all()
    lines = rr.trace(sk, min_spur=4)
    assert any(np.allclose(p[0], p[-1]) for p, _ in lines)      # the circle comes back as a ring


def test_rooms_without_readable_names_come_from_the_walls():
    import cv2
    walls = np.zeros((600, 900), np.uint8)
    cv2.rectangle(walls, (50, 50), (850, 550), 1, 14)          # outer walls
    cv2.line(walls, (450, 50), (450, 550), 1, 14)               # one wall across, with a door gap
    walls[280:330, 443:458] = 0
    cv2.line(walls, (450, 300), (850, 300), 1, 14)
    found_px, outlines = [], []
    lines = rr.unnamed_rooms(walls, (20, 20, 880, 580), [], 0.24, rooms_px=found_px, outlines=outlines)
    assert [t.text for t, _ in lines] == ["ROOM 1", "ROOM 2", "ROOM 3"]
    assert len(outlines) == 3
    n, how = rr.picture_scale([], (20, 20, 880, 580), found_px, 0.24)
    assert n and "estimated" in how
