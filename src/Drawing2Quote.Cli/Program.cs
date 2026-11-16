// Command line: read a drawing and print what was read (as `python -m app.cli`).
//
//   d2q plan.dxf                                  what was read
//   d2q plan.pdf --overlay checked.pdf            also a copy of the PDF showing what was read
//   d2q plan.pdf --pages 5-7,12                   only these pages
//   d2q plan.pdf --json                           the extraction as JSON (as the service's /extract returns it)
//   d2q plan.pdf --json --out plan.json           ... written to a file
using System.Text;
using Drawing2Quote.Reader;

Console.OutputEncoding = Encoding.UTF8;
string? drawing = null, overlay = null, pagesSpec = null, outFile = null, tracedFile = null;
bool json = false, rooms = false;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--json": json = true; break;
        case "--rooms": rooms = true; break;            // each sheet's room outlines (for checking)
        case "--overlay": overlay = args[++i]; break;
        case "--pages": pagesSpec = args[++i]; break;
        case "--out": outFile = args[++i]; break;
        case "--traced": tracedFile = args[++i]; break;      // a picture: the drawing traced from it (PDF)
        case "-h" or "--help":
            Console.WriteLine("usage: d2q drawing [--json] [--out file] [--pages 5-7,12] [--overlay checked.pdf]");
            return 0;
        default: drawing = args[i]; break;
    }
}
if (drawing == null)
{
    Console.Error.WriteLine("usage: d2q drawing [--json] [--out file] [--pages 5-7,12] [--overlay checked.pdf]");
    return 2;
}

Extraction ex;
List<Sheet> sheets;
byte[] data;
try
{
    data = File.ReadAllBytes(drawing);
    Ocr.Progress = (n, total) => Console.Error.WriteLine($"Reading the text of page {n}/{total} from the image (OCR, cached after the first time)...");
    (ex, sheets) = DrawingReader.ExtractWithSheets(Path.GetFileName(drawing), data, pages: Pages(pagesSpec));
}
catch (RejectedFileException exc)
{
    Console.Error.WriteLine($"Rejected: {exc.Message}");
    if (json && outFile != null) File.WriteAllText(outFile, System.Text.Json.JsonSerializer.Serialize(new { rejected = exc.Message }), Encoding.UTF8);
    return 2;
}
if (overlay != null)
{
    if (ex.FileType == "dxf")
    {
        Console.Error.WriteLine("--overlay works on PDF drawings and pictures");
        return 2;
    }
    // a scan or picture: drawn on the traced drawing (the lines and walls as they were traced)
    var basePdF = sheets.Select(s => s.Debug.Pdf).FirstOrDefault(p => p != null) ?? data;
    File.WriteAllBytes(overlay, Overlay.Draw(basePdF, sheets));
    Console.Error.WriteLine($"Marked-up copy written to {overlay}");
}
if (tracedFile != null && sheets.Select(s => s.Debug.Pdf).FirstOrDefault(p => p != null) is { } tracedPdf)
    File.WriteAllBytes(tracedFile, tracedPdf);
if (rooms)
    foreach (var sh in sheets)
        foreach (var r in sh.Rooms)
            Console.Error.WriteLine(r.Poly == null ? $"ROOM {r.Name} | none" :
                FormattableString.Invariant($"ROOM {r.Name} | {r.Poly.Area:F3} | {r.Poly.EnvelopeInternal.MinX:F3} {r.Poly.EnvelopeInternal.MinY:F3} {r.Poly.EnvelopeInternal.MaxX:F3} {r.Poly.EnvelopeInternal.MaxY:F3} | {r.Poly.NumPoints}"));
var text = json ? ex.ToJson() : Show(ex);
if (outFile != null) File.WriteAllText(outFile, text, new UTF8Encoding(false));
else Console.WriteLine(text);
return 0;

static List<int>? Pages(string? spec)
{
    if (string.IsNullOrEmpty(spec)) return null;
    var out_ = new List<int>();
    foreach (var part in spec.Replace(" ", "").Split(','))
    {
        var ab = part.Split('-', 2);
        int a = int.Parse(ab[0]), b = ab.Length > 1 && ab[1].Length > 0 ? int.Parse(ab[1]) : a;
        for (int n = a; n <= b; n++) out_.Add(n);
    }
    return out_;
}

static string Show(Extraction ex)
{
    var out_ = new List<string> { $"{ex.FileName} ({ex.FileType.ToUpperInvariant()})" };
    if (ex.Project.Count > 0) out_.Add("Project: " + string.Join(", ", ex.Project.Select(kv => $"{kv.Key}: {kv.Value}")));
    foreach (var f in ex.Floors)
    {
        out_.Add($"\n{f.Name}");
        foreach (var a in f.Areas)
        {
            var items = a.Items.Count > 0 ? string.Join(", ", a.Items.Select(kv => $"{kv.Key} {kv.Value}")) : "-";
            var area = a.AreaSqft is { } s ? s.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture) : "";
            out_.Add($"  {a.Name,-24} {area,7} sqft  conf {a.Confidence:0.00}  {items}");
        }
    }
    out_.Add($"\nTotal: {ex.Totals.Points} points, {ex.Totals.AreaSqft} sq ft");
    out_.AddRange(ex.Warnings.Select(w => $"! {w}"));
    return string.Join("\n", out_);
}
