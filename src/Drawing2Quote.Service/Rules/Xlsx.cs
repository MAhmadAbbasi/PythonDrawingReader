using System.Text.Json.Nodes;
using ClosedXML.Excel;

namespace Drawing2Quote.Service.Rules;

/// <summary>The product breakdown as an Excel sheet: client name, floors over their areas, a row per product; Qty and
/// Total are formulas (Qty = the areas added up, Total = price x quantity), so the sheet can be edited and adds up
/// again (as breakdown.to_xlsx in the Python service).</summary>
public static class Xlsx
{
    public static byte[] Of(JsonObject b)
    {
        using var wb = new XLWorkbook();
        var ws = wb.Worksheets.Add("Solution");
        var head = XLColor.FromHtml("#D9E1F2");
        var grey = XLColor.FromHtml("#999999");
        const string money = "#,##0";

        void Box(IXLCell c)
        {
            c.Style.Border.OutsideBorder = XLBorderStyleValues.Thin;
            c.Style.Border.OutsideBorderColor = grey;
        }

        const int first = 5;                         // column E: the first area
        var floors = J.Items(b["floors"]).ToList();
        var areas = floors.SelectMany(f => J.Strings(J.Get(f, "areas")).Select(a => (Floor: J.Str(J.Get(f, "name")) ?? "", Area: a))).ToList();
        int last = first + Math.Max(areas.Count, 1) - 1;
        var plan = J.Str(J.Get(b, "plan"));
        var title = ws.Cell(1, 1);
        title.Value = (J.Str(J.Get(b, "client")) ?? "").ToUpperInvariant() + (string.IsNullOrEmpty(plan) ? "" : $" - {Drawing2Quote.Reader.Py.Title(plan)} plan");
        title.Style.Font.Bold = true;
        title.Style.Font.FontSize = 14;
        ws.Range(1, 1, 1, Math.Max(last, 4)).Merge();
        // row 2: each floor over its areas
        int c = first;
        foreach (var f in floors)
        {
            int n = J.Items(J.Get(f, "areas")).Count();
            var cell = ws.Cell(2, c);
            cell.Value = J.Str(J.Get(f, "name"));
            cell.Style.Font.Bold = true;
            cell.Style.Alignment.Horizontal = XLAlignmentHorizontalValues.Center;
            cell.Style.Alignment.Vertical = XLAlignmentVerticalValues.Center;
            cell.Style.Alignment.WrapText = true;
            cell.Style.Fill.BackgroundColor = head;
            if (n > 1) ws.Range(2, c, 2, c + n - 1).Merge();
            c += n;
        }
        // row 3: column headings
        string[] headings = { "Product", "Unit Price", "Qty", "Total" };
        for (int k = 0; k < headings.Length; k++) ws.Cell(3, k + 1).Value = headings[k];
        for (int k = 0; k < areas.Count; k++) ws.Cell(3, first + k).Value = areas[k].Area;
        for (int k = 1; k <= last; k++)
        {
            var cell = ws.Cell(3, k);
            cell.Style.Font.Bold = true;
            cell.Style.Alignment.Horizontal = XLAlignmentHorizontalValues.Center;
            cell.Style.Alignment.Vertical = XLAlignmentVerticalValues.Center;
            cell.Style.Alignment.WrapText = true;
            cell.Style.Fill.BackgroundColor = head;
            Box(cell);
        }

        int r = 4;
        foreach (var row in J.Items(b["rows"]))
        {
            ws.Cell(r, 1).Value = J.Str(J.Get(row, "product"));
            if (J.Num(J.Get(row, "unitPrice")) is { } price)
            {
                ws.Cell(r, 2).Value = price;
                ws.Cell(r, 2).Style.NumberFormat.Format = money;
            }
            int k = 0;
            foreach (var v in J.Items(J.Get(row, "byArea")))
            {
                if (J.Truthy(v)) ws.Cell(r, first + k).Value = J.Num(v) ?? 0;
                k++;
            }
            if (areas.Count > 0 && J.Truthy(J.Get(row, "inAreas")))
                ws.Cell(r, 3).FormulaA1 = $"SUM({XLHelper.GetColumnLetterFromNumber(first)}{r}:{XLHelper.GetColumnLetterFromNumber(last)}{r})";
            else                                      // (package quantity, not all of it placed in an area)
                ws.Cell(r, 3).Value = J.Num(J.Get(row, "quantity")) ?? 0;
            ws.Cell(r, 4).FormulaA1 = $"B{r}*C{r}";
            ws.Cell(r, 4).Style.NumberFormat.Format = money;
            for (int col = 1; col <= last; col++)
            {
                Box(ws.Cell(r, col));
                if (col >= 3) ws.Cell(r, col).Style.Alignment.Horizontal = XLAlignmentHorizontalValues.Center;
            }
            r++;
        }
        int end = r - 1;
        ws.Cell(r, 3).Value = "Subtotal";
        ws.Cell(r, 3).Style.Font.Bold = true;
        if (end >= 4) ws.Cell(r, 4).FormulaA1 = $"SUM(D4:D{end})";
        else ws.Cell(r, 4).Value = 0;
        ws.Cell(r, 4).Style.NumberFormat.Format = money;
        int sub = r++;
        ws.Cell(r, 1).Value = "Installation charges";
        if (J.Num(J.Get(b, "installation")) is { } inst)
        {
            ws.Cell(r, 2).Value = inst;
            ws.Cell(r, 2).Style.NumberFormat.Format = money;
        }
        ws.Cell(r, 3).Value = 1;
        ws.Cell(r, 4).FormulaA1 = $"B{r}*C{r}";
        ws.Cell(r, 4).Style.NumberFormat.Format = money;
        int instRow = r++;
        ws.Cell(r, 1).Value = "Total";
        ws.Cell(r, 1).Style.Font.Bold = true;
        ws.Cell(r, 4).FormulaA1 = $"D{sub}+D{instRow}";
        ws.Cell(r, 4).Style.NumberFormat.Format = money;
        ws.Cell(r, 4).Style.Font.Bold = true;
        if (!J.Truthy(J.Get(b, "priced")))
        {
            var note = ws.Cell(r + 2, 1);
            note.Value = "Prices missing for some products - enter them in column B and the totals update.";
            note.Style.Font.Italic = true;
            note.Style.Font.FontColor = XLColor.FromHtml("#C00000");
        }

        ws.Column(1).Width = 34;
        ws.Column(2).Width = 12;
        ws.Column(3).Width = 7;
        ws.Column(4).Width = 13;
        for (int k = first; k <= last; k++) ws.Column(k).Width = 11;
        ws.Row(3).Height = 32;
        ws.SheetView.FreezeRows(3);
        ws.SheetView.FreezeColumns(4);
        using var ms = new MemoryStream();
        wb.SaveAs(ms);
        return ms.ToArray();
    }
}
