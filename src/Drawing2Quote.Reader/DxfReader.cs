using System.Text;
using System.Text.RegularExpressions;
using ACadSharp;
using ACadSharp.Entities;
using ACadSharp.IO;
using NetTopologySuite.Geometries;

namespace Drawing2Quote.Reader;

/// <summary>DXF: rooms are closed polylines labelled by the room name inside them; electrical points are block
/// references (INSERT) mapped to point types through the drawing's legend and config/block_mapping.yaml.</summary>
public static class DxfReader
{
    // $INSUNITS -> sq ft per drawing unit²
    private static readonly Dictionary<int, double> UnitSqft = new()
    {
        [1] = 1 / 144.0, [2] = 1.0, [4] = 1 / 92903.04, [5] = 1 / 929.0304, [6] = 10.7639,
    };
    private static readonly Dictionary<int, string> UnitName = new()
    {
        [1] = "inches", [2] = "feet", [4] = "millimetres", [5] = "centimetres", [6] = "metres",
    };
    private static readonly Regex RoomLayer = new("ROOM|AREA|SPACE|ZONE", Py.RX | RegexOptions.IgnoreCase);

    /// <summary>ezdxf's fast_plain_mtext: the MTEXT content without inline formatting codes</summary>
    internal static string PlainMText(string text)
    {
        // caret notation (^J ...)
        text = Regex.Replace(text, @"\^(.)", m => ((char)(((m.Groups[1].Value[0] - 64) % 126 + 126) % 126)).ToString(), RegexOptions.Singleline);
        var chars = new StringBuilder();
        var raw = new Stack<char>(text.Reverse());
        const string oneCharCommands = "PNLlOoKkX";
        while (raw.Count > 0)
        {
            char c = raw.Pop();
            if (c == '\\')
            {
                if (raw.Count == 0) break;
                c = raw.Pop();
                if ("\\{}".Contains(c)) chars.Append(c);
                else if (oneCharCommands.Contains(c))
                {
                    if (c == 'P') chars.Append('\n');
                    else if (c == 'N') chars.Append(' ');
                }
                else
                {
                    // multiple character commands are terminated by ';'
                    bool stacking = c == 'S';
                    char first = c;
                    var search = new Stack<char>(raw.Reverse());
                    var pending = new StringBuilder();
                    bool ok = true;
                    while (c != ';')
                    {
                        if (search.Count == 0) { ok = false; break; }
                        c = search.Pop();
                        if (stacking && c != ';') pending.Append(c);
                    }
                    if (ok)
                    {
                        chars.Append(pending);
                        raw = search;
                    }
                    else
                    {
                        chars.Append(pending);
                        chars.Append('\\').Append(first);
                    }
                }
            }
            else if (c is '{' or '}') { }
            else if (c == '%')
            {
                if (raw.Count > 0 && raw.Peek() == '%')
                {
                    raw.Pop();
                    if (raw.Count > 0)
                    {
                        char code = raw.Pop();
                        string? letter = char.ToLowerInvariant(code) switch { 'c' => "Ø", 'd' => "°", 'p' => "±", '%' => "%", _ => null };
                        if (letter != null) chars.Append(letter);
                        else chars.Append('%').Append('%').Append(code);
                    }
                }
                else chars.Append(c);
            }
            else chars.Append(c);
        }
        return chars.ToString();
    }

    private static List<TextItem> Texts(IEnumerable<Entity> msp)
    {
        var out_ = new List<TextItem>();
        foreach (var e in msp)
        {
            try
            {
                string text;
                double h;
                double px, py;
                if (e is MText mt)
                {
                    text = PlainMText(mt.Value ?? "");
                    h = mt.Height;
                    px = mt.InsertPoint.X; py = mt.InsertPoint.Y;
                }
                else if (e is TextEntity te && e is not AttributeBase)
                {
                    text = te.Value ?? "";
                    h = te.Height;
                    px = te.InsertPoint.X; py = te.InsertPoint.Y;
                    // TEXT aligned by its alignment point
                    if ((int)te.HorizontalAlignment != 0 || (int)te.VerticalAlignment != 0)
                    {
                        px = te.AlignmentPoint.X; py = te.AlignmentPoint.Y;
                    }
                }
                else continue;
                var lines = Py.SplitLines(text);
                for (int i = 0; i < lines.Count; i++)
                    if (lines[i].Trim().Length > 0)
                        out_.Add(new TextItem(lines[i].Trim(), px, py - i * h * 1.5, h != 0 ? h : 1.0));
            }
            catch
            {
            }
        }
        return out_;
    }

    private static List<(Geometry Poly, string Layer)> Polygons(IEnumerable<Entity> msp)
    {
        var out_ = new List<(Geometry, string)>();
        foreach (var e in msp)
        {
            try
            {
                List<Pt> pts;
                bool closed;
                if (e is LwPolyline lw)
                {
                    pts = lw.Vertices.Select(v => new Pt(v.Location.X, v.Location.Y)).ToList();
                    closed = lw.IsClosed;
                }
                else if (e is Polyline2D pl)
                {
                    pts = pl.Vertices.Select(v => new Pt(v.Location.X, v.Location.Y)).ToList();
                    closed = pl.IsClosed;
                }
                else continue;
                if (pts.Count >= 3 && (closed || Math.Sqrt(Math.Pow(pts[0].X - pts[^1].X, 2) + Math.Pow(pts[0].Y - pts[^1].Y, 2)) < 1e-6))
                {
                    Geometry poly = Geo.Polygon(pts);
                    if (!Geo.IsValid(poly)) poly = Geo.Buffer(poly, 0);
                    if (poly.Area > 0 && poly is Polygon) out_.Add((poly, e.Layer?.Name ?? ""));
                }
            }
            catch
            {
            }
        }
        return out_;
    }

    private static double Units(CadDocument doc, IList<Geometry> polys, List<string> log)
    {
        int code = (int)doc.Header.InsUnits;
        if (UnitSqft.TryGetValue(code, out var f))
        {
            log.Add($"Drawing units: {UnitName[code]}");
            return f;
        }
        // not set: guess from the size of the rooms (a 12'x14' room = 168 sq ft)
        double median = polys.Count > 0 ? Py.Median(polys.Select(p => p.Area)) : 0;
        code = 4;
        foreach (var (limit, c) in new[] { (60.0, 6), (2000.0, 2), (60000.0, 1), (1.5e6, 5) })
            if (median < limit) { code = c; break; }
        log.Add($"Drawing units not set in the file - guessed {UnitName[code]} from the room sizes");
        return UnitSqft[code];
    }

    public static (Dictionary<string, string> Project, List<Sheet> Sheets, List<LegendRow> Legend) Read(
        byte[] data, SymbolMapper mapper, List<string> log, List<string> warnings)
    {
        CadDocument doc;
        try
        {
            using var ms = new MemoryStream(data);
            using var reader = new ACadSharp.IO.DxfReader(ms);
            doc = reader.Read();
        }
        catch (Exception exc)
        {
            throw new InvalidOperationException($"The DXF file could not be read: {exc.Message}");
        }
        var msp = doc.Entities.ToList();
        log.Add($"DXF version {doc.Header.Version}, {msp.Count} entities in model space");

        var texts = Texts(msp);
        // Step 1 - title block
        var project = Common.TitleBlock(texts);
        log.Add("Step 1 - project info: " + (project.Count > 0 ? string.Join(", ", project.Select(kv => $"{kv.Key}={kv.Value}")) : "no title block found"));

        // Step 2 - floors
        var floorTitles = texts.Select(t => (Name: Common.FloorName(t.Text), t.X, t.Y)).Where(f => f.Name != null)
                               .Select(f => (f.Name!, f.X, f.Y)).ToList();
        var floorNames = floorTitles.Select(f => f.Item1).Distinct().OrderBy(n => n, StringComparer.Ordinal).ToList();
        log.Add("Step 2 - floors: " + (floorNames.Count > 0 ? string.Join(", ", floorNames) : "no floor titles - one floor"));

        // Step 3 - legend: blocks with a description to their right, under a LEGEND title
        var inserts = msp.OfType<Insert>().ToList();
        var legendMap = new Dictionary<string, (string Desc, string Kind)>();
        var legendHandles = new HashSet<ulong>();
        (double X0, double Y0, double X1, double Y1)? legendBox = null;
        foreach (var title in texts.Where(t => Common.IsLegendTitle(Common.Norm(t.Text))).ToList())
        {
            double h = title.H != 0 ? title.H : 1.0;
            var box = (X0: title.X - 30 * h, Y0: title.Y - 150 * h, X1: title.X + 120 * h, Y1: title.Y + h);
            foreach (var ins in inserts)
            {
                double x = ins.InsertPoint.X, y = ins.InsertPoint.Y;
                if (!(box.X0 <= x && x <= box.X1 && box.Y0 <= y && y <= box.Y1)) continue;
                var cands = texts.Where(t => 0 < t.X - x && t.X - x < 60 * h && Math.Abs(t.Y - y) < 1.5 * Math.Max(t.H, h) && !ReferenceEquals(t, title)).ToList();
                if (cands.Count == 0) continue;
                var desc = Py.MinBy(cands, t => t.X - x);
                var name = ins.Block?.Name ?? "";
                var kind = mapper.Classify(desc.Text, name);
                legendHandles.Add(ins.Handle);
                if (!string.IsNullOrEmpty(kind)) legendMap[name.ToUpperInvariant()] = (desc.Text, kind);
            }
            legendBox = box;
        }
        var legend = legendMap.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                              .Select(kv => new LegendRow { Symbol = kv.Key, Description = kv.Value.Desc, Type = kv.Value.Kind }).ToList();
        log.Add($"Step 3 - legend: {legend.Count} symbol(s)" + (legend.Count > 0 ? ": " + string.Join(", ", legend.Select(l => $"{l.Symbol}={l.Type}")) :
                " (no legend found - block names are matched with config/block_mapping.yaml)"));

        // Step 4 - rooms: the smallest closed polyline around each room name
        var polys = Polygons(msp);
        double sqft = Units(doc, polys.Select(p => p.Poly).ToList(), log);
        var roomLayerPolys = polys.Where(p => RoomLayer.IsMatch(p.Layer ?? "")).ToList();
        var candidates = (roomLayerPolys.Count > 0 ? roomLayerPolys : polys)
            .Where(p => 15 <= p.Poly.Area * sqft && p.Poly.Area * sqft <= 20000).ToList();
        var labelTexts = texts.Where(t => Common.FloorName(t.Text) == null && !Common.IsLegendTitle(Common.Norm(t.Text))
                                          && !(legendBox is { } lb && lb.X0 <= t.X && t.X <= lb.X1 && lb.Y0 <= t.Y && t.Y <= lb.Y1)).ToList();
        var owners = new Dictionary<int, List<TextItem>>();
        foreach (var t in labelTexts)
        {
            var sp = Geo.Point(t.X, t.Y);
            var inside = candidates.Select((c, i) => (c.Poly.Area, i)).Where(c => candidates[c.i].Poly.Covers(sp)).ToList();
            if (inside.Count > 0)
            {
                int i = Py.MinBy(inside, c => new Key2(c.Area, c.i)).i;
                if (!owners.TryGetValue(i, out var l)) owners[i] = l = new List<TextItem>();
                l.Add(t);
            }
        }
        var rooms = new List<Room>();
        for (int i = 0; i < candidates.Count; i++)
        {
            var (poly, _) = candidates[i];
            var own = owners.TryGetValue(i, out var o) ? o : new List<TextItem>();
            var dims = own.Select(t => Common.ParseDimensions(t.Text)).FirstOrDefault(d => d != null);
            var named = own.Where(t => Common.RoomType(t.Text) != null).ToList();
            var other = own.Where(t => Common.ParseDimensions(t.Text) == null && Common.Norm(t.Text).Length >= 3
                                       && Py.FullMatch(@"[\d .]+", Common.Norm(t.Text)) == null).ToList();
            var label = named.Count > 0 ? named[0] : other.Count > 0 ? other[0] : null;
            if (label == null && roomLayerPolys.Count == 0) continue;   // an unlabelled outline (wall, furniture, plot boundary ...)
            var c = Geo.InteriorPoint(poly);
            rooms.Add(new Room(label != null ? label.Text.Trim() : $"Room {rooms.Count + 1}", poly, c.X, c.Y,
                               label != null ? Common.RoomType(label.Text) : null,
                               dims?.Text, dims != null ? dims.Value.Sqft : Py.Round(poly.Area * sqft, 1)));
        }
        log.Add($"Step 4 - rooms: {rooms.Count} found ({candidates.Count} closed outlines checked" +
                (roomLayerPolys.Count > 0 ? ", room layers used" : "") + ")");
        if (rooms.Count == 0)
            warnings.Add("No rooms found: draw each room as a closed polyline with its name inside (a ROOM layer works best).");

        // Step 5 - points: every block reference outside the legend
        var points = new List<PointItem>();
        var unknown = new Dictionary<string, int>();
        foreach (var ins in inserts)
        {
            if (legendHandles.Contains(ins.Handle)) continue;
            var name = ins.Block?.Name ?? "";
            string? kind = legendMap.TryGetValue(name.ToUpperInvariant(), out var lk) ? lk.Kind : null;
            if (string.IsNullOrEmpty(kind))
            {
                var names = new List<string?> { name, ins.Layer?.Name };
                names.AddRange(ins.Attributes.Select(a => a.Value));
                kind = mapper.Classify(names.ToArray());
            }
            if (!string.IsNullOrEmpty(kind)) points.Add(new PointItem(kind, ins.InsertPoint.X, ins.InsertPoint.Y));
            else if (kind == null && !name.StartsWith('*')) unknown[name] = (unknown.TryGetValue(name, out var c) ? c : 0) + 1;
        }
        var counts = points.GroupBy(p => p.Kind).ToDictionary(g => g.Key, g => g.Count());
        log.Add("Step 5 - points: " + (counts.Count > 0 ? string.Join(", ", counts.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key} {kv.Value}")) : "none found"));
        if (unknown.Count > 0)
        {
            var top = string.Join(", ", Py.Sorted(unknown, kv => kv.Value, reverse: true).Take(8).Select(kv => $"{kv.Key} ({kv.Value})"));
            warnings.Add($"{unknown.Values.Sum()} block(s) are not mapped to a point type and were not counted: {top}. " +
                         "Add them to config/block_mapping.yaml if they are electrical points.");
        }
        if (points.Count == 0)
            warnings.Add("No electrical symbols found. Symbols must be blocks (INSERT) - exploded symbols can't be counted.");

        var sheet = new Sheet(floorTitles.Count == 1 ? floorTitles[0].Item1 : "Floor 1")
        {
            Rooms = rooms, Points = points, FloorTitles = floorTitles.Count > 1 ? floorTitles : new(),
        };
        return (project, new List<Sheet> { sheet }, legend);
    }
}
