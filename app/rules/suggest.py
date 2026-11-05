"""The suggested solution for a reviewed drawing: the building size's package (from the POS), with the items the
drawing's points decide replaced by what the drawing needs, filtered by the plan:
  Standard: Standard-only and "both" products - never asks
  Premium:  Premium-only products and "both" products; a "both" product that has a Premium-only product of
            the same group is asked about first (use it / skip it / use the Premium-only product instead)
Pure functions - no I/O."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..extract.common import CONFIG_DIR, norm
from .building_size import choose

PLANS = ("standard", "premium")


def load_rules(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "solution_rules.yaml").read_text(encoding="utf8")) or {}


def _qty(value) -> float | int:
    """A package quantity (decimal in the POS, e.g. 12.5 m of rail): a whole number stays an int"""
    try:
        q = float(value or 0)
    except (TypeError, ValueError):
        return 0
    return int(q) if q.is_integer() else round(q, 3)


def rule_list(rules: dict | None = None) -> list[dict]:
    """The drawing rules a company maps its products to (Solution Packages > Drawing rules): name, the gangs a
    rule has ("" = one list), what it counts - and the Matter gateway (one per area of another company's automation)"""
    rules = rules if rules is not None else load_rules()
    out = []
    for r in rules.get("rules") or []:
        gangs = [str(g) for g in range(int(r.get("max_gang", 3)), 0, -1)] if r.get("rule") == "gangs" else [""]
        out.append({"name": r.get("name") or r.get("rule"), "rule": r.get("rule"), "gangs": gangs,
                    "points": r.get("points") or [], "roomTypes": r.get("room_types") or []})
    out.append({"name": MATTER_GATEWAY, "rule": "other_brand", "gangs": [""], "points": [], "roomTypes": []})
    return out


def _candidates(rule: dict, catalog: dict, by_id: dict, by_name: dict) -> dict[str, list[dict]]:
    """{gang/"": catalogue items, best first}: the company's products for this rule (catalog["ruleItems"], by id);
    a rule may also name items itself (tests, the command line)"""
    name = rule.get("name") or rule.get("rule")
    mapped = (catalog.get("ruleItems") or {}).get(name)
    if mapped is not None:
        return {str(g): [by_id[int(i)] for i in ids if int(i) in by_id] for g, ids in mapped.items()}
    named = rule.get("items") or []
    groups = {str(k): v for k, v in named.items()} if isinstance(named, dict) else {"": named}
    return {g: [by_name[norm(n)] for n in names if norm(n) in by_name] for g, names in groups.items()}


def usable(usage: str, plan: str) -> bool:
    """Can go in the plan at all (a "both" product still needs confirming for Premium)"""
    return usage == "both" or usage == plan


def _areas(extraction: dict):
    for f in extraction.get("floors", []):
        for a in f.get("areas", []):
            yield f, a


def _derive(rule: dict, extraction: dict) -> tuple[bool, dict[str, dict[str, int]], str]:
    """(the drawing has what the rule needs, {gang/"": {area: qty}}, explanation)"""
    points = rule.get("points") or []
    room_types = rule.get("room_types") or []
    mult = float(rule.get("multiplier", 1))
    ratio = float(rule.get("ratio", 1)) or 1
    kind = rule.get("rule")
    per_area: dict[str, dict[str, int]] = {}
    if kind == "gangs":
        return _gangs(rule, extraction)
    if kind == "per_point":
        total = 0
        for f, a in _areas(extraction):
            n = sum(int(a.get("items", {}).get(p, 0)) for p in points)
            if n:
                total += n
                per_area.setdefault("", {})[f"{f['name']} / {a['name']}"] = math.ceil(n * mult / ratio)
        qty = math.ceil(total * mult / ratio)
        # the per-area figures are rounded up one by one; keep the total exact
        if per_area.get("") and sum(per_area[""].values()) != qty:
            per_area[""] = {"All areas": qty}
        return total > 0, per_area, f"{total} {'/'.join(points)} point(s)" + (f" x {mult:g}" if mult != 1 else "") + (f" / {ratio:g}" if ratio != 1 else "")
    if kind == "per_area":
        n = 0
        for f, a in _areas(extraction):
            has_points = points and any(int(a.get("items", {}).get(p, 0)) > 0 for p in points)
            is_room = room_types and a.get("roomType") in room_types
            if has_points or is_room:
                n += 1
                per_area.setdefault("", {})[f"{f['name']} / {a['name']}"] = math.ceil(mult)
        what = " or ".join(filter(None, ["/".join(points) and f"{'/'.join(points)} points", "/".join(room_types)]))
        return n > 0, per_area, f"{n} area(s) with {what}"
    if kind == "per_floor":
        floors = [f for f in extraction.get("floors", []) if f.get("name") != "Roof"
                  and any(a.get("name") != "Unassigned" for a in f.get("areas", []))]
        for f in floors:
            per_area.setdefault("", {})[f["name"]] = math.ceil(mult)
        return len(floors) > 0, per_area, f"{len(floors)} floor(s)"
    return False, {}, f"unknown rule {kind!r}"


def _estimated_loads(area: dict, est: dict) -> int:
    """light/fan/exhaust points an area without any drawn probably has (room type and size)"""
    room = area.get("roomType") or "other"
    sqft = area.get("areaSqft") or 0
    lights = max(1, math.ceil(sqft / float(est.get("sqft_per_light", 60)))) if sqft else         int((est.get("lights") or {}).get(room, (est.get("lights") or {}).get("other", 2)))
    return lights + (room in (est.get("fan") or [])) + (room in (est.get("exhaust_fan") or []))


def _gangs(rule: dict, extraction: dict) -> tuple[bool, dict[str, dict[str, int]], str]:
    """Switch boards for every area: the switches drawn there, else its load points, else estimated loads"""
    top = int(rule.get("max_gang", 3))
    loads = rule.get("points") or []
    switches = rule.get("switch_points") or []
    est = rule.get("estimate")
    per_area: dict[str, dict[str, int]] = {}
    counts = {"switch": [0, 0], "load": [0, 0], "estimate": [0, 0]}     # basis -> [areas, gangs]
    for f, a in _areas(extraction):
        items = a.get("items", {})
        drawn_switches = sum(int(items.get(p, 0)) for p in switches)
        drawn_loads = sum(int(items.get(p, 0)) for p in loads)
        if drawn_switches > 0:
            n, basis = drawn_switches, "switch"
        elif drawn_loads > 0:
            n, basis = drawn_loads, "load"
        elif est is not None and a.get("name") != "Unassigned" and f.get("name") != "Roof":    # no lights/fans drawn
            n, basis = _estimated_loads(a, est), "estimate"
        else:
            continue
        counts[basis][0] += 1
        counts[basis][1] += n
        key = f"{f['name']} / {a['name']}"
        full, rest = divmod(n, top)
        if full:
            per_area.setdefault(str(top), {})[key] = full
        if rest:
            per_area.setdefault(str(rest), {})[key] = 1
    why = []
    if counts["switch"][0]:
        why.append(f"{counts['switch'][1]} switch(es) drawn in {counts['switch'][0]} area(s)")
    if counts["load"][0]:
        why.append(f"{counts['load'][1]} {'/'.join(loads)} point(s) in {counts['load'][0]} area(s)")
    if counts["estimate"][0]:
        why.append(f"about {counts['estimate'][1]} load(s) estimated for {counts['estimate'][0]} area(s) with no lights/fans drawn")
    return bool(per_area), per_area, "; ".join(why) + f" - on switch boards of up to {top}"


MATTER_GATEWAY = "Matter gateway"   # the rule's name; its product is the company's (ruleItems)


def _matter_gateway(other: dict, catalog: dict, lines: list[dict], left_out: list[dict], size: dict | None) -> dict | None:
    """The client already uses another company's automation in some areas: one Matter gateway per such area lets
    those products work with ours. It is the last row, in every plan (it is what the client's own devices need).
    The product is the one the company mapped to the "Matter gateway" rule; without one the row is still shown,
    unlinked, so the quotation lists it to add by hand."""
    areas = [str(a).strip() for a in (other.get("areas") or []) if str(a).strip()]
    areas = list(dict.fromkeys(areas))
    if not areas:
        return None
    company = str(other.get("company") or "").strip() or "another company"
    items = catalog.get("items") or []
    by_id = {i["id"]: i for i in items}
    mapped = _candidates({"name": MATTER_GATEWAY}, catalog, by_id, {}).get("", [])
    item = mapped[0] if mapped else None
    line_info = None
    package_qty = 0
    if item:   # already in the package: shown once, at the end
        for bucket in (lines, left_out):
            for l in [l for l in bucket if l["itemId"] == item["id"]]:
                bucket.remove(l)
                if bucket is lines:
                    package_qty += l["quantity"]
    n = len(areas)
    return {
        "itemId": item["id"] if item else 0,
        "name": item["name"] if item else MATTER_GATEWAY,
        "groupName": item.get("groupName") if item else (line_info or {}).get("groupName"),
        "productLineName": (item or {}).get("productLineName") or MATTER_GATEWAY,
        "usage": (item or {}).get("usage") or (line_info or {}).get("usage") or "none",
        "productId": item.get("productId") if item else None,
        "productName": item.get("productName") if item else None,
        "quantity": n + package_qty,
        "baseline": _qty(((item or {}).get("quantities") or {}).get(str(size["id"]), 0)) if size else 0,
        "source": "other",
        "detail": f"client uses {company} products in {n} area(s) - one Matter gateway per area"
                  + (f" + {package_qty} from the package" if package_qty else ""),
        "areas": {a: 1 for a in areas},
        "otherCompany": company,
    }


def suggest(extraction: dict, catalog: dict, plan: str, choices: dict | None = None,
            size_id: int | None = None, rules: dict | None = None, other_brand: dict | None = None) -> dict:
    """choices: None = not asked yet; else {item id (str/int): "use" | "skip" | id of a Premium-only item to use instead}
    other_brand: {"company": name, "areas": ["Ground Floor / Lounge", ...]} when the client uses another company's
    automation in those areas -> a Matter Gateway row (last)"""
    plan = (plan or "").lower()
    if plan not in PLANS:
        raise ValueError("Choose the Standard or the Premium plan.")
    rules = rules if rules is not None else load_rules()
    items = catalog.get("items") or []
    by_name = {norm(i["name"]): i for i in items}
    size = choose(extraction, catalog.get("sizes") or [], rules.get("building_size") or {}, size_id)
    notes: list[str] = []

    qty: dict[int, int] = {}
    source: dict[int, str] = {}
    detail: dict[int, str] = {}
    areas: dict[int, dict[str, int]] = {}
    serves_same: dict[int, set[int]] = {}   # items of one rule serve the same points (Premium alternatives)
    for i in items:
        base = _qty((i.get("quantities") or {}).get(str(size["id"]), 0)) if size else 0
        qty[i["id"]] = base
        source[i["id"]] = "package"
        detail[i["id"]] = f"{size['name']} package" if size else "no package size"

    # the drawing decides the items it has points for
    by_id_all = {i["id"]: i for i in items}
    for rule in rules.get("rules") or []:
        name = rule.get("name") or rule.get("rule")
        applies, per_area, why = _derive(rule, extraction)
        candidates = _candidates(rule, catalog, by_id_all, by_name)
        if not applies:
            notes.append(f"{name}: the drawing shows none of the points this needs - the package quantity is kept.")
            continue
        for gang, found in candidates.items():
            for i in found:
                serves_same.setdefault(i["id"], set()).update(f["id"] for f in found if f["id"] != i["id"])
            pick = next((i for i in found if usable(i.get("usage", "none"), plan)), None)
            wanted = per_area.get(gang, {})
            for i in found:
                if pick is not None and i["id"] == pick["id"]:
                    qty[i["id"]] = sum(wanted.values())
                    source[i["id"]] = "drawing"
                    detail[i["id"]] = why
                    areas[i["id"]] = wanted
                else:
                    qty[i["id"]] = 0          # replaced by the drawing's figure (or not usable in this plan)
                    source[i["id"]] = "drawing"
                    detail[i["id"]] = why + (f" - {pick['name']} used" if pick else "")
            if pick is None and wanted and found:
                label = f"{gang}-gang " if gang else ""
                other = "premium" if plan == "standard" else "standard"
                if all(i.get("usage") == other for i in found):
                    notes.append(f"{name}: {'/'.join(i['name'] for i in found)} is {other.title()} only - not in a {plan.title()} package.")
                else:
                    notes.append(f"{name}: no {plan.title()} product for {label}{'/'.join(i['name'] for i in found)} - categorize one for "
                                 f"{plan.title()} on the Product Categorization tab (needed: {sum(wanted.values())}).")
        if not any(candidates.values()) and applies:
            notes.append(f"{name}: no product is chosen for this drawing rule - choose them on Solution Packages > Drawing rules.")

    def line(i: dict, q: int, extra: dict | None = None) -> dict:
        out = {"itemId": i["id"], "name": i["name"], "groupName": i.get("groupName"), "productLineName": i.get("productLineName"),
               "usage": i.get("usage", "none"), "productId": i.get("productId"), "productName": i.get("productName"),
               "quantity": q, "baseline": _qty((i.get("quantities") or {}).get(str(size["id"]), 0)) if size else 0,
               "source": source[i["id"]], "detail": detail[i["id"]], "areas": areas.get(i["id"], {})}
        if extra:
            out.update(extra)
        return out

    lines: list[dict] = []
    to_confirm: list[dict] = []
    left_out: list[dict] = []
    by_id = {i["id"]: i for i in items}
    choices = {str(k): v for k, v in choices.items()} if choices is not None else None
    added: dict[int, int] = {}      # Premium-only items chosen instead of a "both" product
    for i in items:
        q = qty[i["id"]]
        if q <= 0:
            continue
        usage = i.get("usage", "none")
        if usage == "none":
            left_out.append(line(i, q, {"reason": "not categorised"}))
        elif plan == "standard":
            (lines if usage in ("standard", "both") else left_out).append(line(i, q, None if usage != "premium" else {"reason": "Premium only"}))
        elif usage == "premium":
            lines.append(line(i, q))
        elif usage == "standard":
            left_out.append(line(i, q, {"reason": "Standard only"}))
        else:   # both, Premium plan
            # Premium-only items that serve the same points (same drawing rule); for package-only items the same group
            if i["id"] in serves_same:
                same = [by_id[x] for x in serves_same[i["id"]]]
            else:
                same = [a for a in items if a["id"] not in serves_same and a.get("groupName") and a.get("groupName") == i.get("groupName")]
            alternatives = [{"itemId": a["id"], "name": a["name"]} for a in same if a.get("usage") == "premium"]
            if not alternatives:    # nothing Premium-only to swap it for: just use it, don't ask
                lines.append(line(i, q))
                continue
            if choices is None:
                to_confirm.append(line(i, q, {"alternatives": alternatives}))
                continue
            choice = str(choices.get(str(i["id"]), "use"))
            if choice == "use":
                lines.append(line(i, q, {"confirmed": True}))
            elif choice.isdigit() and int(choice) in by_id and by_id[int(choice)].get("usage") == "premium":
                added[int(choice)] = added.get(int(choice), 0) + q
                left_out.append(line(i, q, {"reason": f"replaced by {by_id[int(choice)]['name']}"}))
            else:
                left_out.append(line(i, q, {"reason": "you chose not to use it"}))
    for alt_id, q in added.items():
        existing = next((l for l in lines if l["itemId"] == alt_id), None)
        if existing:
            existing["quantity"] += q
            existing["detail"] += f" + {q} instead of a Standard/Premium product"
        else:
            source[alt_id], detail[alt_id] = "choice", "chosen instead of a product ticked for both"
            lines.append(line(by_id[alt_id], q))

    order = {i["id"]: n for n, i in enumerate(items)}
    lines.sort(key=lambda l: order.get(l["itemId"], 0))
    gateway = _matter_gateway(other_brand, catalog, lines, left_out, size) if other_brand and not to_confirm else None
    if gateway:
        lines.append(gateway)
        if gateway["itemId"] == 0:
            notes.append(f"{MATTER_GATEWAY}: choose its product on Solution Packages > Drawing rules, so quotations include it.")
    unlinked = [l for l in lines if l["productId"] is None]
    if unlinked and not to_confirm:
        notes.append(f"{len(unlinked)} item(s) are not in the inventory (not on the price list) - their price is entered on the quotation.")
    return {
        "plan": plan,
        "size": size,
        "needsConfirmation": bool(to_confirm),
        "toConfirm": to_confirm,
        "lines": [] if to_confirm else lines,
        "leftOut": left_out,
        "totals": {"items": len(lines) if not to_confirm else 0,
                   "quantity": sum(l["quantity"] for l in lines) if not to_confirm else 0,
                   "unlinked": len(unlinked) if not to_confirm else 0},
        "notes": notes,
    }
