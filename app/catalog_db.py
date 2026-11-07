"""Reads a company's solution data straight from the POS database (MySQL / MariaDB) - for the command line.
The POS itself sends the same data with each request, so the service never needs the database."""
from __future__ import annotations


def parse_connection(conn: str) -> dict:
    """'Server=..;Port=..;Database=..;User=..;Password=..' (the .NET connection string) -> pymysql arguments"""
    parts = {k.strip().lower(): v.strip() for k, _, v in (p.partition("=") for p in conn.split(";") if "=" in p)}
    return {
        "host": parts.get("server") or parts.get("host") or "localhost",
        "port": int(parts.get("port", 3306)),
        "database": parts.get("database") or parts.get("db"),
        "user": parts.get("user") or parts.get("uid") or parts.get("user id") or parts.get("username"),
        "password": parts.get("password") or parts.get("pwd") or "",
    }


def load_catalog(conn: str, tenant_id: int, plan: str = "standard") -> dict:
    """The company's catalogue for a plan, as the POS sends it (CatalogService.ReaderCatalogAsync): products with
    their family, usage, default SKU and the plan's package quantities, the families, and each drawing rule's products"""
    import pymysql

    db = pymysql.connect(**parse_connection(conn), charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)
    with db, db.cursor() as cur:
        cur.execute("SELECT Id, Name FROM CATL_SIZE WHERE TenantId=%s AND IsDeleted=0 ORDER BY SortOrder, Id", (tenant_id,))
        sizes = [{"id": r["Id"], "name": r["Name"]} for r in cur.fetchall()]
        cur.execute("SELECT Id, Code FROM CATL_TIER WHERE TenantId=%s", (tenant_id,))
        tiers = {r["Code"]: r["Id"] for r in cur.fetchall()}
        cur.execute("SELECT FamilyId, TierId FROM CATL_FMTR WHERE TenantId=%s", (tenant_id,))
        fam_tiers: dict[int, set[int]] = {}
        for r in cur.fetchall():
            fam_tiers.setdefault(r["FamilyId"], set()).add(r["TierId"])

        def usage(family_id) -> str:
            t = fam_tiers.get(family_id, set())
            s, p = tiers.get("standard") in t, tiers.get("premium") in t
            return "both" if s and p else "standard" if s else "premium" if p else "none"

        cur.execute("""
            SELECT c.Id, c.Name, c.Unit, c.FamilyId, f.Name AS FamilyName, g.Name AS GroupName, s.Id AS SkuId, s.ProductName AS SkuName
            FROM CATL_PROD c
            LEFT JOIN CATL_FMLY f ON f.TenantId = c.TenantId AND f.Id = c.FamilyId AND f.IsDeleted = 0
            LEFT JOIN CATL_CATG g ON g.TenantId = f.TenantId AND g.Id = f.GroupId AND g.IsDeleted = 0
            LEFT JOIN INVT_PROD s ON s.TenantId = c.TenantId AND s.CatalogProductId = c.Id AND s.IsDefaultSku = 1 AND s.IsDeleted = 0
            WHERE c.TenantId=%s ORDER BY c.Name""", (tenant_id,))
        rows = cur.fetchall()
        cur.execute("SELECT CatalogProductId, PackageSizeId, Quantity FROM CATL_PKLN WHERE TenantId=%s AND TierId=%s",
                    (tenant_id, tiers.get(plan, 0)))
        qty: dict[int, dict[str, float]] = {}
        for r in cur.fetchall():
            qty.setdefault(r["CatalogProductId"], {})[str(r["PackageSizeId"])] = float(r["Quantity"])
        cur.execute("""SELECT f.Id, f.Name, g.Name AS GroupName FROM CATL_FMLY f
                       JOIN CATL_CATG g ON g.TenantId = f.TenantId AND g.Id = f.GroupId AND g.IsDeleted = 0
                       WHERE f.TenantId=%s AND f.IsDeleted=0 ORDER BY g.SortOrder, f.SortOrder""", (tenant_id,))
        lines = [{"name": r["Name"], "groupName": r["GroupName"], "usage": usage(r["Id"])} for r in cur.fetchall()]
        cur.execute("SELECT RuleName, Gang, CatalogProductId FROM CATL_RULE WHERE TenantId=%s ORDER BY RuleName, Gang, `Rank`", (tenant_id,))
        rule_items: dict[str, dict[str, list[int]]] = {}
        for r in cur.fetchall():
            rule_items.setdefault(r["RuleName"], {}).setdefault(r["Gang"], []).append(r["CatalogProductId"])
    items = [{
        "id": r["Id"], "name": r["Name"], "unit": r["Unit"], "groupName": r["GroupName"], "productLineName": r["FamilyName"],
        "usage": usage(r["FamilyId"]) if r["FamilyName"] else "none",
        "productId": r["SkuId"], "productName": r["SkuName"], "quantities": qty.get(r["Id"], {}),
    } for r in rows]
    return {"sizes": sizes, "items": items, "lines": lines, "ruleItems": rule_items}
