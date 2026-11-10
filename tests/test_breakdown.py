"""The solution's product breakdown per area (the quotation sheet layout) and its Excel file."""
import copy
import io
import json
from pathlib import Path

import openpyxl
import pytest
from fastapi.testclient import TestClient

from app.extract import extract
from app.main import app
from app.rules.breakdown import WHOLE_FLOOR, breakdown, to_xlsx
from app.rules.suggest import suggest
from tests import fixtures

DATA = Path(__file__).parent / "data"
PRICES = {"Smart Panel Q2": 78300, "Smart Curtain Motor 1.5Nm": 33495, "Curtain Rail 1 meter": 9765,
          "Gas Sensor": 18684, "IR Remote Control": 12058, "Smoke Sensor": 15000, "Temp and Humidity Sensor": 9000}


@pytest.fixture(scope="module")
def extraction():
    return extract("sample.dxf", fixtures.make_dxf().read_bytes())


@pytest.fixture(scope="module")
def priced():
    c = json.loads((DATA / "catalog.json").read_text(encoding="utf8"))
    for i in c["items"]:
        if i["name"] in PRICES:
            i["unitPrice"] = PRICES[i["name"]]
    return c


def premium(extraction, catalog, **kw):
    first = suggest(extraction, catalog, "premium")
    choices = {str(c["itemId"]): "use" for c in first["toConfirm"]}
    return suggest(extraction, catalog, "premium", choices, **kw)


def test_breakdown_matches_the_lines(extraction, priced):
    r = premium(extraction, priced, installation=130000)
    b = r["breakdown"]
    assert [f["name"] for f in b["floors"]] == ["Ground Floor", "First Floor"]
    areas = [(f["name"], a) for f in b["floors"] for a in f["areas"]]
    for line, row in zip(r["lines"], b["rows"]):
        assert row["quantity"] == line["quantity"]
        # each area's quantity sits in that area's column
        for key, q in line["areas"].items():
            floor, _, area = key.partition(" / ")
            assert row["byArea"][areas.index((floor, area or WHOLE_FLOOR))] == q
    rail = next(x for x in b["rows"] if x["product"] == "Curtain Rail 1 meter")
    assert rail["total"] == 8 * 9765
    assert b["subtotal"] == sum(x["total"] for x in b["rows"]) and b["total"] == b["subtotal"] + 130000
    assert b["priced"] and b["client"] == "Mr. Ahmed Khan"


def test_breakdown_without_prices(extraction):
    catalog = json.loads((DATA / "catalog.json").read_text(encoding="utf8"))
    b = premium(extraction, catalog)["breakdown"]
    assert not b["priced"] and all(x["total"] is None for x in b["rows"]) and b["total"] == 0


def test_no_breakdown_until_confirmed(extraction, priced):
    assert suggest(extraction, priced, "premium")["breakdown"] is None


def test_xlsx_layout_and_formulas(extraction, priced):
    b = premium(extraction, priced, installation=130000)["breakdown"]
    ws = openpyxl.load_workbook(io.BytesIO(to_xlsx(b))).active
    assert ws["A1"].value.startswith("MR. AHMED KHAN")
    assert [ws.cell(3, k).value for k in range(1, 5)] == ["Product", "Unit Price", "Qty", "Total"]
    assert ws["E2"].value == "Ground Floor"                       # floor over its areas (merged)
    assert any(str(m).startswith("E2:") for m in ws.merged_cells.ranges)
    row = next(r for r in range(4, 20) if ws.cell(r, 1).value == "Curtain Rail 1 meter")
    assert ws.cell(row, 3).value.startswith("=SUM(E") and ws.cell(row, 4).value == f"=B{row}*C{row}"
    labels = [ws.cell(r, 1).value for r in range(4, ws.max_row + 1)]
    assert "Installation charges" in labels and "Total" in labels


def test_api_breakdown_and_xlsx(extraction, priced):
    client = TestClient(app)
    first = client.post("/suggest", json={"extraction": extraction, "catalog": priced, "plan": "premium"}).json()
    assert client.post("/suggest/xlsx", json={"extraction": extraction, "catalog": priced, "plan": "premium"}).status_code == 409
    body = {"extraction": extraction, "catalog": priced, "plan": "premium", "installation": 130000, "client": "Mrs. Sobia Malik",
            "choices": {str(c["itemId"]): "use" for c in first["toConfirm"]}}
    r = client.post("/suggest", json=body).json()
    assert r["breakdown"]["client"] == "Mrs. Sobia Malik" and r["breakdown"]["installation"] == 130000
    x = client.post("/suggest/xlsx", json=body)
    assert x.status_code == 200 and "Mrs. Sobia Malik" in x.headers["content-disposition"]
    assert openpyxl.load_workbook(io.BytesIO(x.content)).active["A1"].value.startswith("MRS. SOBIA MALIK")
