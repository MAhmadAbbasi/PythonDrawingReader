"""Rules that keep room areas right on any drawing: room sizes written with fractions, the scale worked out from
dimension strings, rooms taken into a neighbour's outline, impossible areas, and pictures that are not plans."""
import pymupdf
import pytest
from shapely.geometry import box

from app.extract import RejectedFile, extract
from app.extract import pdf_geometry as geo
from app.extract import pdf_reader as pr
from app.extract.common import SymbolMapper, Text, floor_name, parse_dimensions


def test_sizes_with_fractions():
    assert parse_dimensions("10'-4½\" X 14'-0\"")[1] == 145.2
    assert parse_dimensions("16'-0\" x 14'-7½\"")[1] == 234.0          # (was read as 16 x 14)
    assert parse_dimensions("10'-4 1/2\" x 14'")[1] == 145.2
    assert parse_dimensions("11-7* X7-2\"")[1] == 83.0                 # OCR: inch mark read as "*"
    assert parse_dimensions("11-2 X 10-6\"")[1] == 117.2                # marks left out (was read as 2 x 10'-6")
    assert parse_dimensions("12'x14'")[1] == 168
    assert parse_dimensions("2x4 Sq.mm+ ECC 1x4 Sq.mm") is None          # a cable, not a 2' x 4' room
    assert parse_dimensions("3.6m x 4.2m")[1] == pytest.approx(162.7, 0.1)


def test_names_and_point_words():
    assert floor_name("MEZZANINE FLOOR LIGHTING PLAN") == "Mezzanine Floor"
    assert floor_name("MEZANINE FLOOR SEVERAGE PLAN") == "Mezzanine Floor"
    m = SymbolMapper()
    assert m.classify("LIGHT PLUG 5/10Amp") == "socket_5A"
    assert m.classify("SINGLE SOCKET FOR LAMP") == "socket_5A"
    assert m.classify("TV SOCKET") == "tv_point" and m.classify("INTERNET SOCKET") == "data_point"
    assert m.classify("RJ-45 TELEPHONE OUTLET SIMPLEX") == "data_point"
    assert m.classify("DECORATIVE HANGING LIGHT") == "light"


def _line(x0, y0, x1, y1):
    import numpy as np
    pts = np.array([[x0, y0], [x1, y1]], float)
    return geo.Stroke(pts, 0.5, False, False, False, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), "0,0,0")


def _text(s, x0, y0, x1, y1):
    return Text(s, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0), pymupdf.Rect(x0, y0, x1, y1)


def test_scale_from_dimension_strings():
    # 12 pt per foot: each dimension text sits on its dimension line
    lines, items = [], []
    for k, (txt, ft) in enumerate([("9'", 9), ("2'-6\"", 2.5), ("5'-3\"", 5.25), ("7'", 7), ("4'-0\"", 4),
                                   ("10'-6\"", 10.5), ("3'", 3)]):
        y = 50 + 30 * k
        items.append(_line(100, y, 100 + 12 * ft, y))
        lines.append(_text(txt, 100 + 6 * ft - 5, y - 8, 100 + 6 * ft + 5, y - 1))
    feet, agreed = pr._feet_from_dimensions(lines, items)
    assert feet == pytest.approx(12, rel=0.01) and agreed == 7
    assert pr._feet_from_dimensions(lines[:3], items[:3]) is None          # too few to trust


def test_room_limits():
    assert pr.ROOM_MAX_SQFT["bath"] < pr.ROOM_MAX_SQFT["bedroom"]
    assert "garage" not in pr.ROOM_MAX_SQFT and "lawn" not in pr.ROOM_MAX_SQFT    # open spaces: no limit
    for kind, least in pr.ROOM_MIN_SQFT.items():
        assert least < pr.ROOM_MAX_SQFT.get(kind, 10 ** 6), kind


def test_room_wrapping_round_others_rejected():
    finder = geo.RoomFinder([], pymupdf.Rect(0, 0, 1000, 1000), 10, (2, 30000))
    finder._footprint = None
    xs = [500, 300, 700, 500]
    ys = [100, 500, 500, 600]
    # a U-shaped space round the middle of the house holding three other room names in its box
    u = box(100, 50, 900, 900).difference(box(200, 200, 800, 900))
    import numpy as np
    assert not finder._ok(u, 0, np.array(xs, float), np.array(ys, float), np.array([]), np.array([]))


def test_pictures_that_are_not_plans_rejected():
    # a picture with words and lines but no plan in it (a screenshot, a table)
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for k in range(20):
        page.insert_text((40, 60 + 30 * k), f"{632000 + k} Office Expenses item {k}", fontsize=14)
        page.draw_line((30, 70 + 30 * k), (570, 70 + 30 * k))
    png = page.get_pixmap(dpi=150).tobytes("png")
    with pytest.raises(RejectedFile, match="doesn't look like a floor plan"):
        extract("screenshot.png", png)
