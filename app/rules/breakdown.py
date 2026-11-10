"""The solution as a product breakdown per area (the layout of the quotation sheets: a row per product with its unit
price, quantity and total, and a column per area grouped under its floor), and the same as an Excel file."""
from __future__ import annotations

import io

WHOLE_FLOOR = "Whole floor"


def breakdown(result: dict, extraction: dict, catalog: dict | None = None, client: str | None = None,
              installation: float | None = None) -> dict:
    """result: suggest()'s answer; extraction: the drawing it was made from.
    Prices: catalog items' "unitPrice" (sent by the POS); a line without one has no price or total.
    installation: installation charges (a row of its own, added to the total)"""
    prices = {i["id"]: i.get("unitPrice") for i in (catalog or {}).get("items") or []}
    lines = [l for l in result.get("lines") or [] if l.get("quantity")]

    # the columns: every area of the drawing that a product goes to, floor by floor in the drawing's order
    used = {k for l in lines for k in (l.get("areas") or {})}
    floors = []
    for f in extraction.get("floors") or []:
        cols = [a["name"] for a in f.get("areas") or [] if f"{f['name']} / {a['name']}" in used]
        if f["name"] in used:
            cols.append(WHOLE_FLOOR)
        if cols:
            floors.append({"name": f["name"], "areas": cols})
    keys = [(f["name"], a) for f in floors for a in f["areas"]]

    rows = []
    for l in lines:
        per = l.get("areas") or {}
        cells = [per.get(f if a == WHOLE_FLOOR else f"{f} / {a}") or None for f, a in keys]
        qty = l["quantity"]
        price = l.get("unitPrice", prices.get(l.get("itemId")))
        rows.append({"itemId": l.get("itemId"), "product": l.get("productName") or l["name"], "unitPrice": price,
                     "quantity": qty, "total": round(price * qty, 2) if price is not None else None,
                     # (all of it in the areas listed - else part is the package's, kept in Qty only)
                     "byArea": cells, "inAreas": sum(c or 0 for c in cells) == qty})
    subtotal = round(sum(r["total"] or 0 for r in rows), 2)
    return {
        "client": client or (extraction.get("project") or {}).get("client") or extraction.get("fileName") or "",
        "plan": result.get("plan"),
        "floors": floors,
        "rows": rows,
        "subtotal": subtotal,
        "installation": installation,
        "total": round(subtotal + (installation or 0), 2),
        "priced": all(r["unitPrice"] is not None for r in rows),
    }


def to_xlsx(b: dict) -> bytes:
    """The breakdown as an Excel sheet: client name, floors over their areas, a row per product; Qty and Total are
    formulas (Qty = the areas added up, Total = price x quantity), so the sheet can be edited and adds up again."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter as col

    wb = Workbook()
    ws = wb.active
    ws.title = "Solution"
    thin = Side(style="thin", color="999999")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    bold = Font(bold=True)
    head = PatternFill("solid", fgColor="D9E1F2")
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)
    money = "#,##0"

    first = 5                                    # column E: the first area
    areas = [(f["name"], a) for f in b["floors"] for a in f["areas"]]
    last = first + max(len(areas), 1) - 1
    ws.cell(1, 1, b["client"].upper() + (f" - {b['plan'].title()} plan" if b.get("plan") else "")).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(last, 4))
    # row 2: each floor over its areas
    c = first
    for f in b["floors"]:
        n = len(f["areas"])
        cell = ws.cell(2, c, f["name"])
        cell.font, cell.alignment, cell.fill = bold, centre, head
        if n > 1:
            ws.merge_cells(start_row=2, start_column=c, end_row=2, end_column=c + n - 1)
        c += n
    # row 3: column headings
    for k, title in enumerate(["Product", "Unit Price", "Qty", "Total"], start=1):
        ws.cell(3, k, title)
    for k, (_, a) in enumerate(areas):
        ws.cell(3, first + k, a)
    for k in range(1, last + 1):
        cell = ws.cell(3, k)
        cell.font, cell.alignment, cell.fill, cell.border = bold, centre, head, box

    r = 4
    for row in b["rows"]:
        ws.cell(r, 1, row["product"])
        if row["unitPrice"] is not None:
            ws.cell(r, 2, row["unitPrice"]).number_format = money
        for k, v in enumerate(row["byArea"]):
            if v:
                ws.cell(r, first + k, v)
        if areas and row["inAreas"]:
            ws.cell(r, 3, f"=SUM({col(first)}{r}:{col(last)}{r})")
        else:                                    # (package quantity, not all of it placed in an area)
            ws.cell(r, 3, row["quantity"])
        ws.cell(r, 4, f"=B{r}*C{r}").number_format = money
        for k in range(1, last + 1):
            ws.cell(r, k).border = box
            if k >= 3:
                ws.cell(r, k).alignment = Alignment(horizontal="center")
        r += 1
    end = r - 1
    ws.cell(r, 3, "Subtotal").font = bold
    ws.cell(r, 4, f"=SUM(D4:D{end})" if end >= 4 else 0).number_format = money
    sub = r
    r += 1
    ws.cell(r, 1, "Installation charges")
    if b.get("installation") is not None:
        ws.cell(r, 2, b["installation"]).number_format = money
    ws.cell(r, 3, 1)
    ws.cell(r, 4, f"=B{r}*C{r}").number_format = money
    inst = r
    r += 1
    ws.cell(r, 1, "Total").font = bold
    ws.cell(r, 4, f"=D{sub}+D{inst}").number_format = money
    ws.cell(r, 4).font = bold
    if not b.get("priced"):
        ws.cell(r + 2, 1, "Prices missing for some products - enter them in column B and the totals update.").font = Font(italic=True, color="C00000")

    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 7
    ws.column_dimensions["D"].width = 13
    for k in range(first, last + 1):
        ws.column_dimensions[col(k)].width = 11
    ws.row_dimensions[3].height = 32
    ws.freeze_panes = "E4"
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
