"""Room and symbol detection on a CAD-style PDF (tests/cad_fixture.py): rooms traced from double-line walls with
door openings, symbols made of several strokes and turned on the walls."""
import math

import numpy as np
import pymupdf
import pytest

from app.extract import extract, extract_with_sheets
from app.extract import pdf_geometry as geo
from app.extract.overlay import draw
from tests import cad_fixture


def counts(ex):
    return {a["name"]: a["items"] for f in ex["floors"] for a in f["areas"]}


@pytest.fixture(scope="module")
def cad(tmp_path_factory):
    return cad_fixture.make_cad_pdf(tmp_path_factory.mktemp("cad") / "plan.pdf").read_bytes()


def test_cad_pdf_exact_counts(cad):
    ex = extract("plan.pdf", cad)
    got = counts(ex)
    for room, items in cad_fixture.EXPECTED.items():
        assert got[room] == items, room
    assert got["Unassigned"] == cad_fixture.UNASSIGNED
    assert set(got) == set(cad_fixture.EXPECTED) | {"Unassigned"}       # no stray "rooms" from symbol letters
    assert any("8 room(s) (8 traced from walls)" in line for line in ex["log"])
    areas = {a["name"]: a["areaSqft"] for a in ex["floors"][0]["areas"]}
    assert areas["MASTER BED ROOM"] == 224                               # written 14'x16'
    # traced: the clear space inside the walls (wall 0.75 ft thick)
    assert areas["KITCHEN"] == pytest.approx((12 - 0.75) * (10 - 0.75), rel=0.03)
    assert areas["PASSAGE"] == pytest.approx((40 - 0.75) * (5 - 0.75), rel=0.03)


def test_cad_pdf_without_scale(tmp_path):
    data = cad_fixture.make_cad_pdf(tmp_path / "plan.pdf", scale_text=False).read_bytes()
    ex = extract("plan.pdf", data)
    got = counts(ex)
    for room, items in cad_fixture.EXPECTED.items():
        assert got[room] == items, room
    areas = {a["name"]: a["areaSqft"] for a in ex["floors"][0]["areas"]}
    # scale worked out from the written room sizes (wall centre to centre), so within ~15% of the clear space
    assert areas["KITCHEN"] == pytest.approx(11.25 * 9.25, rel=0.15)
    assert any("no scale written" in line for line in ex["log"])


def test_cad_pdf_without_legend(tmp_path):
    data = cad_fixture.make_cad_pdf(tmp_path / "plan.pdf", legend=False).read_bytes()
    ex = extract("plan.pdf", data)
    assert ex["totals"]["points"] == 0
    assert {a["name"] for a in ex["floors"][0]["areas"]} == set(cad_fixture.EXPECTED)
    assert any("repeated symbol shape" in w for w in ex["warnings"])


def test_overlay(cad):
    ex, sheets = extract_with_sheets("plan.pdf", cad)
    out = pymupdf.open(stream=draw(cad, sheets), filetype="pdf")
    assert out.page_count == 1
    assert "LOUNGE" in out[0].get_text() and "224 sq ft" in out[0].get_text()


def _stroke(points, curve=False, filled=False):
    pts = np.array(points, float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return geo.Stroke(pts, 0.5, filled, False, curve, (lo[0], lo[1], hi[0], hi[1]))


def _light(cx, cy, turn=0.0):
    t = np.linspace(0, 2 * math.pi, 33)
    circle = _stroke(np.c_[cx + 4 * np.cos(t), cy + 4 * np.sin(t)], curve=True)
    cross = [_stroke([(cx - 4 * math.cos(a), cy - 4 * math.sin(a)), (cx + 4 * math.cos(a), cy + 4 * math.sin(a))])
             for a in (turn + math.pi / 4, turn + 3 * math.pi / 4)]
    return [circle] + cross


def test_shape_ignores_rotation_and_position():
    a = geo.describe(_light(0, 0))
    b = geo.describe(_light(100, 50, turn=0.3))
    plain = geo.describe(_light(0, 0)[:1])                      # circle without the cross
    assert geo.shape_distance(a, b) < 0.15
    assert geo.shape_distance(a, plain) > geo.MATCH_LIMIT


def test_symbols_grouped_from_separate_strokes():
    items = _light(0, 0) + _light(40, 0) + [_stroke([(-100, 20), (100, 20)])]   # two lights and a long wall line
    found = geo.find_symbols(items, max_size=12, gap=1)
    assert len(found) == 2 and {round(s.x) for s in found} == {0, 40}


def test_same_floor_on_several_sheets_is_merged():
    from shapely.geometry import box as sbox
    from app.extract.common import PointItem, Room, Sheet, assemble

    def sheet(points):
        rooms = [Room("BATH", sbox(0, 0, 10, 10), 5, 5, "bath", None, 100.0),
                 Room("BATH", sbox(20, 0, 30, 10), 25, 5, "bath", None, 100.0),
                 Room("LOUNGE", sbox(40, 0, 60, 20), 50, 10, "lounge", None, 400.0)]
        return Sheet("Ground Floor", rooms, [PointItem(k, x, y) for k, x, y in points])

    lighting = sheet([("light", 2, 2), ("light", 3, 3), ("light", 22, 2), ("light", 45, 5)])
    power = sheet([("socket_5A", 2, 2), ("light", 3, 3), ("socket_5A", 22, 2), ("socket_5A", 23, 2)])
    floors = assemble([lighting, power], [], [])
    areas = {a["name"]: a["items"] for a in floors[0]["areas"]}
    assert len(floors) == 1 and len(floors[0]["areas"]) == 3          # two baths stay two, nothing doubled
    assert areas["BATH 1"] == {"light": 2, "socket_5A": 1}              # the light on both sheets counts once
    assert areas["BATH 2"] == {"light": 1, "socket_5A": 2}
    assert areas["LOUNGE"] == {"light": 1}
