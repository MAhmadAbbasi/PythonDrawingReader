import json
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.extract import RejectedFile, extract
from app.extract.common import parse_dimensions, room_type, SymbolMapper
from app.main import app
from app.rules.building_size import choose
from app.rules.suggest import suggest
from tests import fixtures

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def dxf():
    return fixtures.make_dxf().read_bytes()


@pytest.fixture(scope="session")
def pdf():
    return fixtures.make_pdf().read_bytes()


@pytest.fixture(scope="session")
def catalog():
    return json.loads((DATA / "catalog.json").read_text(encoding="utf8"))


def counts(ex):
    return {a["name"]: a["items"] for f in ex["floors"] for a in f["areas"]}


# ── upload rules ──
@pytest.mark.parametrize("name,data,words", [
    ("plan.dwg", b"AC1027" + b"\0" * 100, "Save As -> DXF"),
    ("photo.jpg", b"\xff\xd8\xff\xe0" + b"\0" * 100, "Photos and images"),
    ("plan.png", b"\x89PNG\r\n\x1a\n" + b"\0" * 100, "Photos and images"),
    ("plan.heic", b"\0" * 100, "Photos and images"),
    ("plan.docx", b"PK\x03\x04", "not supported"),
    ("plan.dxf", b"hello", "not a valid DXF"),
    ("empty.pdf", b"", "empty"),
])
def test_rejected_files(name, data, words):
    with pytest.raises(RejectedFile, match=words.replace(">", ".")):
        extract(name, data)


def test_too_big():
    with pytest.raises(RejectedFile, match="maximum is 50 MB"):
        extract("big.dxf", b"0\nSECTION\n" + b" " * (50 * 1024 * 1024))


def test_scanned_pdf_rejected():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), 0)
    page.insert_image(page.rect, pixmap=pix)
    with pytest.raises(RejectedFile, match="appears to be scanned"):
        extract("scan.pdf", doc.tobytes())


def test_password_pdf_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "x")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(RejectedFile, match="password-protected"):
        extract("locked.pdf", data)


# ── reading ──
def test_dxf_exact_counts(dxf):
    ex = extract("sample.dxf", dxf)
    got = counts(ex)
    for room, items in fixtures.EXPECTED.items():
        assert got[room] == items, room
    assert got["Unassigned"] == fixtures.UNASSIGNED
    assert [f["name"] for f in ex["floors"]] == ["Ground Floor", "First Floor"]
    assert {a["name"] for a in ex["floors"][1]["areas"]} == {"BED ROOM", "GUEST ROOM"}
    assert ex["project"]["client"] == "Mr. Ahmed Khan" and ex["project"]["plotSize"] == "10 Marla"
    master = next(a for a in ex["floors"][0]["areas"] if a["name"] == "MASTER BED ROOM")
    assert master["areaSqft"] == 196 and master["dimensions"] == "14'x14'" and master["confidence"] == 1.0
    bed = next(a for a in ex["floors"][1]["areas"] if a["name"] == "BED ROOM")
    assert bed["areaSqft"] == 168 and bed["dimensions"] is None       # from the polyline
    assert any("XYZ1" in w for w in ex["warnings"]) and not any("SOFA" in w for w in ex["warnings"])
    assert len(ex["legend"]) == 6


def test_pdf_exact_counts(pdf):
    ex = extract("sample.pdf", pdf)
    got = counts(ex)
    for room, items in fixtures.EXPECTED.items():
        assert got[room] == items, room
    assert got["Unassigned"] == fixtures.UNASSIGNED
    bed = next(a for a in ex["floors"][1]["areas"] if a["name"] == "BED ROOM")
    assert bed["areaSqft"] == 168     # from the rectangle and SCALE 1:72


def test_text_rules():
    assert parse_dimensions("12'-6\" x 14'-0\"")[1] == 175.0
    assert parse_dimensions("12 x 14")[1] == 168
    assert parse_dimensions("3.6m x 4.2m")[1] == pytest.approx(162.7, 0.1)
    assert parse_dimensions("LOUNGE") is None
    assert room_type("MASTER BED ROOM") == "bedroom" and room_type("Wash Room") == "bath" and room_type("T.V. LOUNGE") == "lounge"
    m = SymbolMapper()
    assert m.classify("SKT15A") == "socket_15A" and m.classify("SKT5A") == "socket_5A" and m.classify("EXH-FAN") == "exhaust_fan"
    assert m.classify("SOFA_3") == "" and m.classify("XYZ1") is None


# ── building size ──
def test_building_size(catalog):
    sizes, cfg = catalog["sizes"], {"sqft_per_marla": 225, "floor_coverage": 0.8, "bhk_max_sqft": 1800}
    assert choose({"project": {"plotSize": "1 Kanal"}, "floors": []}, sizes, cfg)["name"] == "1 Kanal"
    # largest floor 1800 sq ft / 0.8 = 2250 sq ft plot = 10 Marla
    two_floors = {"project": {}, "floors": [{"name": n, "areas": [{"name": "LOUNGE", "areaSqft": 1800, "roomType": "lounge", "items": {}}]}
                                            for n in ("Ground Floor", "First Floor")]}
    assert choose(two_floors, sizes, cfg)["name"] == "10 Marla"
    flat = {"project": {}, "floors": [{"name": "Floor 1", "areas": [{"name": f"BED {i}", "areaSqft": 150, "roomType": "bedroom", "items": {}} for i in range(2)]}]}
    assert choose(flat, sizes, cfg)["name"] == "2 BHK"
    assert choose(flat, sizes, cfg, override_id=sizes[-1]["id"])["name"] == sizes[-1]["name"]


# ── plans ──
def lines(result):
    return {l["name"]: l["quantity"] for l in result["lines"]}


def test_standard_never_asks(dxf, catalog):
    r = suggest(extract("sample.dxf", dxf), catalog, "standard")
    assert not r["needsConfirmation"] and not r["toConfirm"]
    got = lines(r)
    # 7+10+5+4+3 light/fan points and the unassigned light, boards of up to 3
    assert got["Piano Switch 3 Gang"] == 8 and got["Piano Switch 2 Gang"] == 1 and got["Piano Switch 1 Gang"] == 4
    assert got["IR Remote Control"] == 3 and got["Gas Sensor"] == 1
    assert got["Smart Curtain Motor 1.5Nm"] == 2 and got["Curtain Rail 1 meter"] == 8
    assert got["Smart Lock i5"] == 1 and got["Smart Panel Q2"] == 2          # 10 Marla package
    assert "Smoke Sensor" not in got and "Smart Lock i9 Pro" not in got        # Premium only
    usage = {l["usage"] for l in r["lines"]}
    assert usage <= {"standard", "both"}


def test_premium_always_asks_first(dxf, catalog):
    ex = extract("sample.dxf", dxf)
    first = suggest(ex, catalog, "premium")
    assert first["needsConfirmation"] and first["lines"] == []
    asked = {c["name"]: c for c in first["toConfirm"]}
    assert set(asked) == {"Smart Panel Q2", "Smart Curtain Motor 1.5Nm", "Curtain Rail 1 meter", "Gas Sensor", "IR Remote Control"}
    assert [a["name"] for a in asked["Smart Panel Q2"]["alternatives"]] == ["Smart Panel Hpad X Pro"]
    assert asked["Gas Sensor"]["alternatives"] == []                           # smoke sensor serves other points

    choices = {str(c["itemId"]): "use" for c in first["toConfirm"]}
    choices[str(asked["Smart Panel Q2"]["itemId"])] = str(asked["Smart Panel Q2"]["alternatives"][0]["itemId"])
    choices[str(asked["Gas Sensor"]["itemId"])] = "skip"
    r = suggest(ex, catalog, "premium", choices)
    got = lines(r)
    assert not r["needsConfirmation"]
    assert got["Smart Panel Hpad X Pro"] == 2 and "Smart Panel Q2" not in got   # replaced
    assert "Gas Sensor" not in got and got["Smoke Sensor"] == 1 and got["Temp and Humidity Sensor"] == 2
    assert "Piano Switch 3 Gang" not in got                                     # Standard only
    assert any("no Premium item for 3-gang" in n for n in r["notes"])
    assert {l["usage"] for l in r["lines"]} <= {"premium", "both"}


def test_rules_round_up_and_room_types(catalog):
    ex = {"project": {"plotSize": "5 Marla"}, "floors": [{"name": "Ground Floor", "areas": [
        {"name": "KITCHEN", "roomType": "kitchen", "areaSqft": 100, "items": {"light": 1}},
        {"name": "LOUNGE", "roomType": "lounge", "areaSqft": 200, "items": {"light": 4, "curtain": 3}},
        {"name": "PASSAGE", "roomType": "corridor", "areaSqft": 60, "items": {}},
    ]}]}
    rules = {"rules": [
        {"name": "rail", "rule": "per_point", "points": ["curtain"], "multiplier": 1, "ratio": 2, "items": ["Curtain Rail 1 meter"]},
        {"name": "gas", "rule": "per_area", "room_types": ["kitchen"], "items": ["Gas Sensor"]},
        {"name": "motion", "rule": "per_area", "room_types": ["corridor"], "items": ["Motion Sensor"]},
        {"name": "sw", "rule": "gangs", "points": ["light"], "max_gang": 3, "items": {"3": ["Piano Switch 3 Gang"], "1": ["Piano Switch 1 Gang"]}},
    ]}
    got = lines(suggest(ex, catalog, "standard", rules=rules))
    assert got["Curtain Rail 1 meter"] == 2         # 3 / 2 rounded up
    assert got["Gas Sensor"] == 1 and got["Motion Sensor"] == 1
    assert got["Piano Switch 3 Gang"] == 1 and got["Piano Switch 1 Gang"] == 2   # lounge 4 = 3 + 1, kitchen 1


def test_totals_without_prices(dxf, catalog):
    cat = json.loads(json.dumps(catalog))
    cat["items"][0]["productId"], cat["items"][0]["productName"] = 99, "Linked product"
    r = suggest(extract("sample.dxf", dxf), cat, "standard")
    assert r["totals"]["quantity"] == sum(l["quantity"] for l in r["lines"])
    assert r["totals"]["unlinked"] == sum(1 for l in r["lines"] if l["productId"] is None)
    assert all("unitPrice" not in l and "lineTotal" not in l for l in r["lines"])
    assert "amount" not in r["totals"]


# ── service ──
def test_api(dxf, catalog):
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    bad = client.post("/extract", files={"file": ("plan.dwg", b"AC1027xxxx")})
    assert bad.status_code == 422 and "DXF" in bad.json()["detail"]
    ex = client.post("/extract", files={"file": ("sample.dxf", dxf)}).json()
    r = client.post("/suggest", json={"extraction": ex, "catalog": catalog, "plan": "premium"}).json()
    assert r["needsConfirmation"]
    assert client.post("/suggest", json={"extraction": ex, "catalog": catalog, "plan": "gold"}).status_code == 422


# ── background jobs (long readings: printed PDFs are read with OCR) ──
def test_jobs(dxf):
    import time as _t
    client = TestClient(app)
    r = client.post("/jobs", files={"file": ("plan.dwg", b"AC1027", "application/octet-stream")})
    assert r.status_code == 422 and "DXF" in r.json()["detail"]        # refused straight away
    job = client.post("/jobs", files={"file": ("sample.dxf", dxf, "application/dxf")}).json()["jobId"]
    for _ in range(100):
        s = client.get(f"/jobs/{job}").json()
        if s["status"] in ("done", "failed"):
            break
        _t.sleep(0.1)
    assert s["status"] == "done" and s["result"]["totals"]["points"] == 51
    assert client.get("/jobs/nope").status_code == 404
