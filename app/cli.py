"""Command line: read a drawing and print the suggested solution.

  python -m app.cli plan.dxf                                   just the extraction
  python -m app.cli plan.pdf --overlay checked.pdf             also a copy of the PDF showing what was read
  python -m app.cli plan.pdf --pages 5-7,12                    only these pages
  python -m app.cli plan.dxf --plan standard --catalog catalog.json
  python -m app.cli plan.dxf --plan premium --db "Server=..;Database=..;User=..;Password=.." --tenant 3 --choices all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import RejectedFile, extract_with_sheets
from .rules.suggest import suggest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drawing2quote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drawing")
    ap.add_argument("--plan", choices=["standard", "premium"])
    ap.add_argument("--catalog", help="the POS solution data as JSON")
    ap.add_argument("--db", help="POS database connection string (reads the solution data directly)")
    ap.add_argument("--tenant", type=int, help="company id in the POS database")
    ap.add_argument("--size", type=int, help="building size id (default: worked out from the drawing)")
    ap.add_argument("--choices", default="ask", help="Premium 'both' products: ask (list them), all, none")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--pages", help="PDF pages to read, e.g. 5-7,12 (default: all)")
    ap.add_argument("--overlay", help="PDF / picture: write a copy of the drawing with the rooms and symbols found marked on it")
    ap.add_argument("--xlsx", help="with --plan: write the solution's product breakdown per area as an Excel sheet")
    ap.add_argument("--installation", type=float, help="installation charges (on the product breakdown)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    path = Path(args.drawing)
    try:
        data = path.read_bytes()
        from .extract import ocr
        ocr.progress = lambda n, total: print(f"Reading the text of page {n}/{total} from the image (OCR, cached "
                                             "after the first time)...", file=sys.stderr, flush=True)
        ex, sheets = extract_with_sheets(path.name, data, pages=_pages(args.pages))
    except RejectedFile as exc:
        print(f"Rejected: {exc}", file=sys.stderr)
        return 2
    if args.overlay:
        if ex["fileType"] == "dxf":
            ap.error("--overlay works on PDF drawings and pictures")
        from .extract.overlay import draw
        # a scan or picture: drawn on the traced drawing (the lines and walls as they were traced)
        base = next((s.debug["pdf"] for s in sheets if s.debug and "pdf" in s.debug), data)
        Path(args.overlay).write_bytes(draw(base, sheets))
        print(f"Marked-up copy written to {args.overlay}", file=sys.stderr)
    if not args.plan:
        print(json.dumps(ex, indent=2) if args.json else _show_extraction(ex))
        return 0
    if args.catalog:
        catalog = json.loads(Path(args.catalog).read_text(encoding="utf8"))
    elif args.db and args.tenant:
        from .catalog_db import load_catalog
        catalog = load_catalog(args.db, args.tenant, args.plan)
    else:
        ap.error("--plan needs --catalog, or --db with --tenant")
    choices = None
    if args.plan == "premium" and args.choices in ("all", "none"):
        first = suggest(ex, catalog, "premium", None, args.size)
        choices = {str(c["itemId"]): "use" if args.choices == "all" else "skip" for c in first["toConfirm"]}
    result = suggest(ex, catalog, args.plan, choices, args.size, installation=args.installation)
    if args.xlsx and result.get("breakdown"):
        from .rules.breakdown import to_xlsx
        Path(args.xlsx).write_bytes(to_xlsx(result["breakdown"]))
        print(f"Product breakdown written to {args.xlsx}", file=sys.stderr)
    print(json.dumps(result, indent=2) if args.json else _show_extraction(ex) + "\n\n" + _show_result(result))
    return 0


def _pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _show_extraction(ex: dict) -> str:
    out = [f"{ex['fileName']} ({ex['fileType'].upper()})"]
    if ex["project"]:
        out.append("Project: " + ", ".join(f"{k}: {v}" for k, v in ex["project"].items()))
    for f in ex["floors"]:
        out.append(f"\n{f['name']}")
        for a in f["areas"]:
            items = ", ".join(f"{k} {v}" for k, v in a["items"].items()) or "-"
            out.append(f"  {a['name']:<24} {str(a['areaSqft'] or ''):>7} sqft  conf {a['confidence']:.2f}  {items}")
    out.append(f"\nTotal: {ex['totals']['points']} points, {ex['totals']['areaSqft']} sq ft")
    out += [f"! {w}" for w in ex["warnings"]]
    return "\n".join(out)


def _show_result(r: dict) -> str:
    out = [f"{r['plan'].title()} plan - {r['size']['name']} ({r['size']['reason']})" if r["size"] else r["plan"].title()]
    if r["needsConfirmation"]:
        out.append("Confirm these products ticked for both Standard and Premium (use --choices all|none):")
        out += [f"  ? {c['name']} x {c['quantity']}" + (f"  (or {', '.join(a['name'] for a in c['alternatives'])})" if c["alternatives"] else "")
                for c in r["toConfirm"]]
        return "\n".join(out)
    for l in r["lines"]:
        product = l["productName"] or "(no product linked)"
        out.append(f"  {l['quantity']:>4} x {l['name']:<34} {l['source']:<8} {product:<30}   {l['detail']}")
    t = r["totals"]
    out.append(f"  Total {t['items']} products, {t['quantity']} pieces" + (f" ({t['unlinked']} not linked to a product)" if t["unlinked"] else ""))
    out += [f"- {n}" for n in r["notes"]]
    if r.get("breakdown"):
        out.append("\n" + _show_breakdown(r["breakdown"]))
    return "\n".join(out)


def _show_breakdown(b: dict) -> str:
    """The product breakdown per area as a text table (columns numbered, the areas listed above it)"""
    areas = [(f["name"], a) for f in b["floors"] for a in f["areas"]]
    out = ["Product breakdown per area" + "".join(f"\n  [{k + 1}] {f} / {a}" for k, (f, a) in enumerate(areas))]
    out.append(f"  {'Product':<30} {'Price':>9} {'Qty':>4} {'Total':>11}  " + " ".join(f"{k + 1:>3}" for k in range(len(areas))))
    for r in b["rows"]:
        price = f"{r['unitPrice']:,.0f}" if r["unitPrice"] is not None else "-"
        total = f"{r['total']:,.0f}" if r["total"] is not None else "-"
        out.append(f"  {r['product'][:30]:<30} {price:>9} {r['quantity']:>4} {total:>11}  "
                   + " ".join(f"{v or '':>3}" for v in r["byArea"]))
    if b.get("installation") is not None:
        out.append(f"  {'Installation charges':<30} {b['installation']:>9,.0f} {1:>4} {b['installation']:>11,.0f}")
    out.append(f"  {'Total':<30} {'':>9} {'':>4} {b['total']:>11,.0f}" + ("" if b["priced"] else "   (prices missing for some products)"))
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
