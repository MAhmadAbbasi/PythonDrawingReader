"""Reads a company's solution data straight from the POS database (MySQL / MariaDB) - for the command line.
The POS itself sends the same data with each request, so the service never needs the database."""
from __future__ import annotations


def _usage(std: int | None, prem: int | None) -> str:
    if std is None:
        return "none"
    return "both" if std and prem else "standard" if std else "premium" if prem else "none"


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


def load_catalog(conn: str, tenant_id: int) -> dict:
    import pymysql

    db = pymysql.connect(**parse_connection(conn), charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)
    with db, db.cursor() as cur:
        cur.execute("SELECT Id, Name FROM PackageSizes WHERE TenantId=%s AND IsDeleted=0 ORDER BY SortOrder, Id", (tenant_id,))
        sizes = [{"id": r["Id"], "name": r["Name"]} for r in cur.fetchall()]
        cur.execute("""
            SELECT i.Id, i.Name, i.ProductId, l.Name AS LineName, l.InStandard, l.InPremium, g.Name AS GroupName,
                   p.ProductName
            FROM PackageItems i
            LEFT JOIN SolutionProductLines l ON l.Id = i.ProductLineId AND l.IsDeleted = 0
            LEFT JOIN SolutionGroups g ON g.Id = l.GroupId AND g.IsDeleted = 0
            LEFT JOIN Products p ON p.Id = i.ProductId AND p.IsDeleted = 0
            WHERE i.TenantId=%s AND i.IsDeleted=0 ORDER BY i.SortOrder, i.Id""", (tenant_id,))
        rows = cur.fetchall()
        cur.execute("""SELECT q.PackageItemId, q.PackageSizeId, q.Quantity FROM PackageQuantities q
                       WHERE q.TenantId=%s AND q.IsDeleted=0 AND q.Quantity > 0""", (tenant_id,))
        qty: dict[int, dict[str, int]] = {}
        for r in cur.fetchall():
            qty.setdefault(r["PackageItemId"], {})[str(r["PackageSizeId"])] = r["Quantity"]
        cur.execute("""SELECT l.Name, g.Name AS GroupName, l.InStandard, l.InPremium FROM SolutionProductLines l
                       JOIN SolutionGroups g ON g.Id = l.GroupId AND g.IsDeleted = 0
                       WHERE l.TenantId=%s AND l.IsDeleted=0 ORDER BY g.SortOrder, l.SortOrder""", (tenant_id,))
        lines = [{"name": r["Name"], "groupName": r["GroupName"], "usage": _usage(r["InStandard"], r["InPremium"])} for r in cur.fetchall()]
    items = [{
        "id": r["Id"], "name": r["Name"], "groupName": r["GroupName"], "productLineName": r["LineName"],
        "usage": _usage(r["InStandard"], r["InPremium"]) if r["LineName"] else "none",
        "productId": r["ProductId"] if r["ProductName"] else None, "productName": r["ProductName"], "quantities": qty.get(r["Id"], {}),
    } for r in rows]
    return {"sizes": sizes, "items": items, "lines": lines}
