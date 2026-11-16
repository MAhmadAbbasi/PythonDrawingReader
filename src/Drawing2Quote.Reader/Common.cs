using System.Text.RegularExpressions;
using NetTopologySuite.Geometries;
using YamlDotNet.RepresentationModel;

namespace Drawing2Quote.Reader;

/// <summary>A line of text on the drawing: its centre (x, y) and height</summary>
public sealed class TextItem
{
    public string Text;
    public double X, Y, H;

    public TextItem(string text, double x, double y, double h = 1.0)
    {
        Text = text; X = x; Y = y; H = h;
    }

    public override string ToString() => $"{Text} @ ({X:0.#}, {Y:0.#})";
}

/// <summary>Shared pieces of the DXF and PDF readers: text rules, symbol mapping and putting points into rooms</summary>
public static class Common
{
    public static string ConfigDir = Path.Combine(AppContext.BaseDirectory, "config");

    private static readonly Regex NonAlnum = new(@"[^A-Za-z0-9]+", Py.RX);
    private static readonly Regex LetterDigit = new(@"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", Py.RX);
    private static readonly Regex Spaces = new(@"\s+", Py.RX);

    /// <summary>Upper case, symbols -> spaces, a space between letters and digits ("SKT15A" -> "SKT 15 A")</summary>
    public static string Norm(string? text)
    {
        var s = NonAlnum.Replace(text ?? "", " ").ToUpperInvariant();
        s = LetterDigit.Replace(s, " ");
        return Spaces.Replace(s, " ").Trim();
    }

    // ── floors ──
    public static readonly (string Pattern, string Name)[] FloorPatterns =
    {
        (@"\bLOWER GROUND\b", "Lower Ground Floor"),
        (@"\bUPPER GROUND\b", "Upper Ground Floor"),
        (@"\bBASEMENT\b", "Basement"),
        (@"\bGROUND FLOOR\b|\bG ?F PLAN\b", "Ground Floor"),
        (@"\bMEZZ?ANINE\b|\bMEZ\b", "Mezzanine Floor"),
        (@"\bFIRST FLOOR\b|\b1 ?ST FLOOR\b", "First Floor"),
        (@"\bSECOND FLOOR\b|\b2 ?ND FLOOR\b", "Second Floor"),
        (@"\bTHIRD FLOOR\b|\b3 ?RD FLOOR\b", "Third Floor"),
        (@"\bROOF\b|\bTERRACE FLOOR\b|\bMUMTY\b", "Roof"),
    };

    public static string? FloorName(string text)
    {
        var n = Norm(text);
        if (n.Length > 60) return null;
        foreach (var (pattern, name) in FloorPatterns)
            // written together only for long words ("BASEMENTFLOOR"; not "ROOF" inside "WEATHERPROOF")
            if (Matches(new[] { pattern }, n, 7)) return name;
        return null;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(string, int), Regex?> JoinedCache = new();

    /// <summary>The pattern for text read with its spaces lost ("MASTERBEDROOM"); null for short abbreviations,
    /// which would match inside other words</summary>
    internal static Regex? Joined(string pattern, int minLetters = 4) => JoinedCache.GetOrAdd((pattern, minLetters), key =>
    {
        var alts = new List<string>();
        foreach (var alt in key.Item1.Split('|'))          // each alternative on its own ("ROOF|TERRACE FLOOR")
        {
            var q = alt.Replace(@"\b", "").Replace(" ?", "").Replace(" ", "");
            if (Regex.Replace(q, "[^A-Z]", "", Py.RX).Length >= key.Item2) alts.Add(q);
        }
        return alts.Count > 0 ? new Regex(string.Join("|", alts), Py.RX) : null;
    });

    /// <summary>n (normalised text) matches one of the patterns - also when its words are written together</summary>
    public static bool Matches(IEnumerable<string> patterns, string n, int minLetters = 4)
    {
        var list = patterns as IList<string> ?? patterns.ToList();
        if (list.Any(p => Py.Search(p, n))) return true;
        var joined = n.Replace(" ", "");
        return list.Any(p => Joined(p, minLetters) is { } j && j.IsMatch(joined));
    }

    // ── rooms ──
    public static readonly (string Kind, string[] Patterns)[] RoomTypes =
    {
        ("dressing", new[] { @"\bDRESS" }),
        ("bath", new[] { @"\bBATH", @"WASH ?ROOM", @"\bTOILET", @"\bW ?C\b", @"\bPOWDER", @"\bLAV", @"\bPWD\b" }),
        ("bedroom", new[] { @"BED ?R(OO)?M", @"\bBED\b", @"\bMASTER\b", @"GUEST ROOM", @"\bKIDS?\b", @"\bB ?R\b" }),
        ("kitchen", new[] { @"KITCHEN", @"PANTRY", @"\bKIT\b" }),
        ("dining", new[] { @"DINING" }),
        ("lounge", new[] { @"LOUNGE", @"LIVING", @"DRAWING", @"\bT ?V\b", @"FAMILY", @"SITTING", @"\bHALL\b" }),
        ("corridor", new[] { @"CORRIDOR", @"PASSAGE", @"LOBBY", @"FOYER", @"ENTRANCE", @"GALLERY" }),
        ("stairs", new[] { @"\bSTAIR", @"\bLIFT\b", @"\bELEVATOR\b", @"\bMUMTY\b" }),
        ("store", new[] { @"\bSTORE", @"\bSTORAGE\b" }),
        ("garage", new[] { @"GARAGE", @"\bPORCH\b", @"CAR PARK", @"PARKING" }),
        ("laundry", new[] { @"LAUNDRY", @"\bWASH AREA\b", @"\bUTILITY\b" }),
        ("servant", new[] { @"SERVANT", @"\bMAID\b", @"\bDRIVER\b", @"\bGUARD\b", @"\bSTAFF\b", @"\bQUARTERS?\b" }),
        ("study", new[] { @"\bSTUDY\b", @"\bOFFICE\b", @"LIBRARY", @"\bPRAYER\b" }),
        ("lawn", new[] { @"\bLAWN\b", @"\bGARDEN\b", @"\bCOURT ?YARD\b", @"\bPATIO\b" }),
        ("balcony", new[] { @"BALCONY", @"\bTERRACE\b", @"\bVERANDAH?\b" }),
        ("other", new[] { @"^ROOM \d+$" }),      // a space enclosed by walls whose name could not be read (pictures)
    };

    public static string? RoomType(string text)
    {
        var n = Norm(text);
        if (n.Length == 0 || n.Length > 40) return null;
        foreach (var (kind, patterns) in RoomTypes)
            if (patterns.Any(p => Py.Search(p, n))) return kind;
        foreach (var (kind, patterns) in RoomTypes)     // words written together ("MASTERBEDROOM")
            if (Matches(patterns, n)) return kind;
        return null;
    }

    // 12'x14'  12'-6" x 14'-0"  12 x 14  3.6m x 4.2m
    private const string Ft = @"(\d+(?:\.\d+)?)\s*(?:'|FT|FEET)?\s*(?:-?\s*(\d+(?:\.\d+)?)\s*(?:""|''|IN))?";
    private static readonly Regex DimFeet = new(@"(?<![\d.])" + Ft + @"\s*[xX×*]\s*" + Ft + @"(?![\d.])", Py.RX);
    private static readonly Regex DimMetre = new(@"(\d+(?:\.\d+)?)\s*M\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*M\b", Py.RX | RegexOptions.IgnoreCase);
    private static readonly Regex CableSize = new(@"SQ\W{0,2}MM|\bMM\b|MM2|\bCORE\b", Py.RX);
    private static readonly Regex StarInch = new(@"(\d\s*-\s*\d+(?:\.\d+)?)\s*\*(?=\s*[xX×])", Py.RX);
    private static readonly Regex NoMarks = new(@"(?<![\d.])(\d{1,3})\s*['′]?\s*-\s*(\d{1,2}(?:\.\d+)?)(?![\d.""″'])", Py.RX);

    /// <summary>(the dimension text, area in sq ft) of a room size like 12'x14' - null if it isn't one</summary>
    public static (string Text, double Sqft)? ParseDimensions(string? text)
    {
        var t = (text ?? "").Trim();
        if (t.Length == 0 || t.Length > 40) return null;
        if (CableSize.IsMatch(t.ToUpperInvariant())) return null;     // a cable size ("2x4 Sq.mm"), not a room
        var m = DimMetre.Match(t);
        if (m.Success)
            return (t, Py.Round(Py.ParseFloat(m.Groups[1].Value) * Py.ParseFloat(m.Groups[2].Value) * 10.7639, 1));
        // fractions of an inch as decimals: 4½" / 4 1/2" -> 4.5"
        var s = t;
        foreach (var (frac, dec) in new[] { ("½", ".5"), ("¼", ".25"), ("¾", ".75") })
            s = Py.Rx(@"(\d)\s*" + frac).Replace(s, "${1}" + dec);
        foreach (var (frac, dec) in new[] { ("1/2", ".5"), ("1/4", ".25"), ("3/4", ".75") })
            s = Py.Rx(@"(\d)[\s-]+" + frac + @"(?=\s*[""″'])").Replace(s, "${1}" + dec);
        // (OCR: an inch mark read as "*" - 11-7* X7-2")
        s = StarInch.Replace(s, "${1}\"");
        // feet-inches with the marks left out: 11-2 X 10-6" -> 11'-2" X 10'-6" (not 2 x 10'-6")
        s = NoMarks.Replace(s, "${1}'-${2}\"");
        m = DimFeet.Match(s.ToUpperInvariant());
        if (!m.Success) return null;
        double a = Py.ParseFloat(m.Groups[1].Value) + (m.Groups[2].Success ? Py.ParseFloat(m.Groups[2].Value) : 0) / 12;
        double b = Py.ParseFloat(m.Groups[3].Value) + (m.Groups[4].Success ? Py.ParseFloat(m.Groups[4].Value) : 0) / 12;
        if (!(2 <= a && a <= 200 && 2 <= b && b <= 200)) return null;
        return (t, Py.Round(a * b, 1));
    }

    private static readonly Regex PlotSizeRx = new(@"(\d+(?:\.\d+)?)\s*(MARLA|KANAL|BHK)\b", Py.RX | RegexOptions.IgnoreCase);

    public static string? PlotSize(string? text)
    {
        var m = PlotSizeRx.Match(text ?? "");
        if (!m.Success) return null;
        var unit = m.Groups[2].Value.ToUpperInvariant();
        var g = m.Groups[1].Value;
        var number = g.Contains('.') ? g.TrimEnd('0').TrimEnd('.') : g;
        return $"{number} {(unit == "BHK" ? "BHK" : Py.Title(unit))}";
    }

    // ── title block ──
    public static readonly (string Key, string[] Patterns)[] TitleFields =
    {
        ("projectName", new[] { @"^PROJECT( NAME| TITLE)?\b", @"^TITLE\b", @"^NAME OF PROJECT\b" }),
        ("client", new[] { @"^CLIENT\b", @"^OWNER\b", @"^CUSTOMER\b" }),
        ("drawingNumber", new[] { @"^(DRAWING|DWG|SHEET) ?(NO|NUMBER|#)\b" }),
        ("scale", new[] { @"^SCALE\b" }),
        ("date", new[] { @"^DATE\b" }),
        ("architect", new[] { @"^ARCHITECT\b", @"^ENGINEER\b", @"^DESIGNED BY\b", @"^CONSULTANT\b", @"^DRAWN BY\b" }),
    };

    internal static bool IsTitleField(string n) => TitleFields.Any(f => f.Patterns.Any(p => Py.Search(p, n)));

    /// <summary>Project info: "PROJECT: Smith House" in one text, or the label with its value next to it</summary>
    public static Dictionary<string, string> TitleBlock(IList<TextItem> texts)
    {
        var info = new Dictionary<string, string>();
        foreach (var t in texts)
        {
            var n = Norm(t.Text);
            foreach (var (key, patterns) in TitleFields)
            {
                if (info.ContainsKey(key) || !patterns.Any(p => Py.Search(p, n))) continue;
                var raw = t.Text.Trim();
                var value = raw.Contains(':') ? raw[(raw.IndexOf(':') + 1)..].Trim() : "";
                if (value.Length == 0)
                {
                    // "SCALE 1:100" - the rest of the text after the label
                    Match? m = null;
                    foreach (var pat in patterns)
                        if ((m = Py.MatchStart(pat.Replace("^", @"^\s*"), raw.ToUpperInvariant())) != null) break;
                    value = m != null ? raw[(m.Index + m.Length)..].Trim(' ', ':', '-') : "";
                }
                if (value.Length == 0)
                {
                    // the nearest text to the right on the same line, or just below
                    TextItem? best = null;
                    double dist = double.PositiveInfinity;
                    foreach (var o in texts)
                    {
                        if (ReferenceEquals(o, t) || o.Text.Trim().Length == 0) continue;
                        double dx = o.X - t.X, dy = Math.Abs(o.Y - t.Y), d;
                        if (0 < dx && dx < 60 * Math.Max(t.H, 1e-6) && dy < 0.8 * Math.Max(t.H, o.H)) d = dx;
                        else if (Math.Abs(dx) < 10 * t.H && 0 < dy && dy < 2.5 * Math.Max(t.H, o.H)) d = 1000 * t.H + dy;
                        else continue;
                        if (d < dist) { best = o; dist = d; }
                    }
                    value = best != null ? best.Text.Trim() : "";
                }
                if (value.Length > 0 && !IsTitleField(Norm(value)))
                    info[key] = value.Length > 150 ? value[..150] : value;
            }
        }
        foreach (var t in texts)
        {
            var size = PlotSize(t.Text);
            if (size != null) info.TryAdd("plotSize", size);
        }
        return info;
    }

    public static readonly string LegendTitle = @"^(ELECTRICAL )?(LEGENDS?|KEY|SYMBOLS?|SYMBOL LIST)\b";

    public static bool IsLegendTitle(string normalised) => Py.MatchStart(LegendTitle, normalised) != null;

    // ── putting it together ──
    internal static string NearestFloor(Sheet sheet, double x, double y)
    {
        if (sheet.Views.Count > 0)                       // the drawing it is drawn in
        {
            var inside = sheet.Views.Where(v => v.Box.X0 <= x && x <= v.Box.X1 && v.Box.Y0 <= y && y <= v.Box.Y1).ToList();
            if (inside.Count > 0)
                return Py.MinBy(inside, v => (v.Box.X1 - v.Box.X0) * (v.Box.Y1 - v.Box.Y0)).Name;
            return Py.MinBy(sheet.Views, v => Math.Pow(Math.Max(Math.Max(v.Box.X0 - x, 0), x - v.Box.X1), 2)
                                             + Math.Pow(Math.Max(Math.Max(v.Box.Y0 - y, 0), y - v.Box.Y1), 2)).Name;
        }
        if (sheet.FloorTitles.Count == 0) return sheet.DefaultFloor;
        return Py.MinBy(sheet.FloorTitles, f => (f.X - x) * (f.X - x) + (f.Y - y) * (f.Y - y)).Name;
    }

    private static Dictionary<string, int> SortedItems(Dictionary<string, int> counts) =>
        counts.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(kv => kv.Key, kv => kv.Value);

    private static readonly Regex NotLetters = new("[^A-Z]", Py.RX);

    /// <summary>Adds a room read on one sheet. The same floor drawn on several sheets (lighting layout, power layout
    /// ...) gives the same room more than once: it is merged into the room of that name read on another sheet (the
    /// nearest one, when the name is used twice), keeping the highest count of each point type - a point drawn on
    /// two sheets is one point.</summary>
    private static void Add(List<Area> areas, Area n)
    {
        foreach (var a in areas) a.Sheets ??= new HashSet<int> { a.SheetNo };
        var others = areas.Where(a => !a.Sheets!.Contains(n.SheetNo)).ToList();
        var same = others.Where(a => a.Name == n.Name).ToList();
        if (same.Count == 0 && n.Name != "Unassigned")
        {
            // read differently on another sheet (OCR: "STAFF QUARTER," / "STAFF QUARTER") at the same place
            var key = NotLetters.Replace(n.Name.ToUpperInvariant(), "");
            same = others.Where(a => a.Name != "Unassigned"
                                     && Py.SequenceRatio(key, NotLetters.Replace(a.Name.ToUpperInvariant(), "")) >= 0.75
                                     && Math.Sqrt(Math.Pow(a.Pos.X - n.Pos.X, 2) + Math.Pow(a.Pos.Y - n.Pos.Y, 2)) < 60).ToList();
        }
        if (same.Count == 0)
        {
            areas.Add(n);
            return;
        }
        var old = Py.MinBy(same, a => Math.Pow(a.Pos.X - n.Pos.X, 2) + Math.Pow(a.Pos.Y - n.Pos.Y, 2));
        old.Sheets!.Add(n.SheetNo);
        foreach (var (k, v) in n.Items)
            old.Items[k] = Math.Max(old.Items.TryGetValue(k, out var c) ? c : 0, v);
        old.Items = SortedItems(old.Items);
        old.AreaSqft ??= n.AreaSqft;
        old.Dimensions ??= n.Dimensions;
        old.RoomType ??= n.RoomType;
        old.Confidence = Math.Max(old.Confidence, n.Confidence);
    }

    /// <summary>Rooms and their point counts per floor, plus an "Unassigned" area for points outside every room</summary>
    public static List<Floor> Assemble(IList<Sheet> sheets, List<string> log, List<string> warnings)
    {
        var floors = new Dictionary<string, List<Area>>();
        List<Area> FloorOf(string name)
        {
            if (!floors.TryGetValue(name, out var l)) floors[name] = l = new List<Area>();
            return l;
        }
        foreach (var sheet in sheets)          // every floor plan drawn is listed, even when no room in it could be read
            foreach (var v in sheet.Views) FloorOf(v.Name);
        for (int sheetNo = 0; sheetNo < sheets.Count; sheetNo++)
        {
            var sheet = sheets[sheetNo];
            var areas = sheet.Rooms.Select(r => (Room: r, Counts: new Dictionary<string, int>(), Floor: NearestFloor(sheet, r.X, r.Y))).ToList();
            var polys = areas.Select((a, i) => (i, a.Room.Poly)).Where(p => p.Poly != null).ToList();
            var labels = areas.Select((a, i) => (i, a.Room)).Where(p => p.Room.Poly == null).ToList();
            // a room known only by its label takes the points near it (within a typical room's size)
            var sizes = polys.Select(p => Math.Sqrt(p.Poly!.Area)).OrderBy(v => v).ToList();
            double reach = sizes.Count > 0 ? sizes[sizes.Count / 2] : double.PositiveInfinity;
            var unassigned = new Dictionary<string, Dictionary<string, int>>();
            foreach (var p in sheet.Points)
            {
                var sp = new Point(p.X, p.Y);
                var inside = polys.Where(q => q.Poly!.Covers(sp)).Select(q => (Key: q.Poly!.Area, q.i)).ToList();
                if (inside.Count == 0 && sheet.Snap != 0)
                    inside = polys.Select(q => (Key: q.Poly!.Distance(sp), q.i)).Where(q => q.Key <= sheet.Snap).ToList();
                int? idx;
                if (inside.Count > 0)
                    idx = Py.MinBy(inside, q => new Key2(q.Key, q.i)).i;
                else
                {
                    var near = labels.Select(l => (D: Math.Sqrt(Math.Pow(l.Room.X - p.X, 2) + Math.Pow(l.Room.Y - p.Y, 2)), l.i))
                                     .Where(q => q.D <= reach).ToList();
                    idx = near.Count > 0 ? Py.MinBy(near, q => new Key2(q.D, q.i)).i : null;
                }
                if (idx == null)
                {
                    var floor = NearestFloor(sheet, p.X, p.Y);
                    if (!unassigned.TryGetValue(floor, out var u)) unassigned[floor] = u = new Dictionary<string, int>();
                    u[p.Kind] = (u.TryGetValue(p.Kind, out var c) ? c : 0) + 1;
                }
                else
                {
                    var counts = areas[idx.Value].Counts;
                    counts[p.Kind] = (counts.TryGetValue(p.Kind, out var c) ? c : 0) + 1;
                }
            }
            foreach (var (room, counts, floor) in areas)
            {
                double conf = 1.0 - sheet.ConfidencePenalty;
                if (room.Poly == null) conf -= 0.3;
                if (room.AreaSqft == null) conf -= 0.15;
                if (room.Kind == null) conf -= 0.2;
                if (counts.Count == 0) conf -= 0.1;
                Add(FloorOf(floor), new Area
                {
                    Name = room.Name, RoomType = room.Kind, AreaSqft = room.AreaSqft, Dimensions = room.Dimensions,
                    Items = SortedItems(counts), Confidence = Py.Round(Math.Max(0.05, Math.Min(1.0, conf)), 2),
                    SheetNo = sheetNo, Pos = (room.X, room.Y),
                });
            }
            foreach (var (floor, counts) in unassigned)
                Add(FloorOf(floor), new Area
                {
                    Name = "Unassigned", Items = SortedItems(counts), Confidence = 0.0, SheetNo = sheetNo, Pos = (0.0, 0.0),
                });
        }
        foreach (var (floor, areas) in floors)
        {
            var un = areas.FirstOrDefault(a => a.Name == "Unassigned");
            if (un != null)
                warnings.Add($"{floor}: {un.Items.Values.Sum()} point(s) are outside every room - they are in " +
                             "\"Unassigned\"; move them to the right room.");
        }
        var order = FloorPatterns.Select(f => f.Name).ToList();
        var result = new List<Floor>();
        var names = floors.Keys.Select(f => (f, o: order.IndexOf(f) is var i && i >= 0 ? i : order.Count))
                          .OrderBy(x => x.o).ThenBy(x => x.f, StringComparer.Ordinal).Select(x => x.f);
        foreach (var name in names)
        {
            var areas = floors[name];
            // same room name twice on a floor (two bedrooms labelled BED ROOM): number them
            var seen = new Dictionary<string, int>();
            foreach (var a in areas) seen[a.Name] = (seen.TryGetValue(a.Name, out var c) ? c : 0) + 1;
            var dup = new Dictionary<string, int>();
            foreach (var a in areas)
            {
                if (seen[a.Name] > 1 && a.Name != "Unassigned")
                {
                    dup[a.Name] = (dup.TryGetValue(a.Name, out var c) ? c : 0) + 1;
                    a.Name = $"{a.Name} {dup[a.Name]}";
                }
            }
            foreach (var a in areas)
                if (a.Name != "Unassigned" && a.Confidence < 0.7)
                    warnings.Add($"{name} / {a.Name}: low confidence ({Py.Repr(a.Confidence)}) - check its counts.");
            if (areas.Count == 0)
                warnings.Add($"{name}: drawn, but no room in it could be read - add its rooms by hand.");
            double sum = 0;
            foreach (var a in areas) sum += a.AreaSqft ?? 0;
            result.Add(new Floor { Name = name, Areas = areas, AreaSqft = Py.Round(sum, 1) });
            log.Add($"{name}: {areas.Count(a => a.Name != "Unassigned")} room(s), {areas.Sum(a => a.Items.Values.Sum())} point(s)");
        }
        return result;
    }

    public static Totals Totals(IList<Floor> floors)
    {
        var items = new Dictionary<string, int>();
        double sqft = 0.0;
        foreach (var f in floors)
            foreach (var a in f.Areas)
            {
                sqft += a.AreaSqft ?? 0;
                foreach (var (k, v) in a.Items) items[k] = (items.TryGetValue(k, out var c) ? c : 0) + v;
            }
        return new Totals { AreaSqft = Py.Round(sqft, 1), Items = SortedItems(items), Points = items.Values.Sum() };
    }
}

/// <summary>Block name / legend description -> point type (light, fan, socket_5A ...), from block_mapping.yaml</summary>
public sealed class SymbolMapper
{
    private readonly Dictionary<string, string> _blocks = new();
    private readonly List<(string Name, Regex[] Patterns)> _types = new();
    private readonly List<(string Name, Regex[] Patterns)> _joined = new();
    private readonly List<Regex> _ignore = new();
    public IReadOnlyList<string> TypesList { get; }

    public SymbolMapper(string? path = null)
    {
        var file = path ?? Path.Combine(Common.ConfigDir, "block_mapping.yaml");
        var yaml = new YamlStream();
        using (var reader = new StringReader(File.ReadAllText(file)))
            yaml.Load(reader);
        var root = yaml.Documents.Count > 0 ? yaml.Documents[0].RootNode as YamlMappingNode : null;
        YamlNode? Get(string key) => root != null && root.Children.TryGetValue(new YamlScalarNode(key), out var v) ? v : null;
        if (Get("blocks") is YamlMappingNode blocks)
            foreach (var (k, v) in blocks.Children)
                _blocks[((YamlScalarNode)k).Value!.ToUpperInvariant()] = ((YamlScalarNode)v).Value ?? "";
        var names = new List<string>();
        if (Get("types") is YamlMappingNode types)
            foreach (var (k, v) in types.Children)
            {
                var name = ((YamlScalarNode)k).Value!;
                var patterns = v is YamlSequenceNode seq ? seq.Children.Select(c => ((YamlScalarNode)c).Value ?? "").ToList() : new List<string>();
                _types.Add((name, patterns.Select(p => new Regex(p, Py.RX)).ToArray()));
                _joined.Add((name, patterns.Select(p => Common.Joined(p)).Where(j => j != null).Select(j => j!).ToArray()));
                names.Add(name);
            }
        if (Get("ignore") is YamlSequenceNode ignore)
            foreach (var w in ignore.Children)
                _ignore.Add(new Regex(@"\b" + Regex.Escape(((YamlScalarNode)w).Value ?? "") + @"\b", Py.RX));
        TypesList = names;
    }

    /// <summary>The point type, "" when it is something to ignore (furniture, doors ...), null when unknown</summary>
    public string? Classify(params string?[] names)
    {
        foreach (var name in names)
            if (!string.IsNullOrEmpty(name) && _blocks.TryGetValue(name.ToUpperInvariant(), out var kind)) return kind;
        foreach (var name in names)
        {
            var n = Common.Norm(name);
            if (n.Length == 0) continue;
            foreach (var (kind, patterns) in _types)
                if (patterns.Any(p => p.IsMatch(n))) return kind;
        }
        foreach (var name in names)                       // words written together ("CEILINGFAN")
        {
            var n = Common.Norm(name).Replace(" ", "");
            foreach (var (kind, patterns) in _joined)
                if (n.Length > 0 && patterns.Any(p => p.IsMatch(n))) return kind;
        }
        foreach (var name in names)
        {
            var n = Common.Norm(name);
            if (n.Length > 0 && _ignore.Any(p => p.IsMatch(n))) return "";
        }
        return null;
    }
}

public sealed class Room
{
    public string Name;
    public Geometry? Poly;        // null: only the label position is known (points go to the nearest label)
    public double X, Y;
    public string? Kind;
    public string? Dimensions;
    public double? AreaSqft;

    public Room(string name, Geometry? poly, double x, double y, string? kind = null, string? dimensions = null, double? areaSqft = null)
    {
        Name = name; Poly = poly; X = x; Y = y; Kind = kind; Dimensions = dimensions; AreaSqft = areaSqft;
    }
}

public sealed class PointItem
{
    public string Kind;
    public double X, Y;

    public PointItem(string kind, double x, double y)
    {
        Kind = kind; X = x; Y = y;
    }
}

/// <summary>A drawing found on a sheet (floor plan, elevation, section ...) for the "sections" list</summary>
public sealed class ViewInfo
{
    public int Page;
    public string? Name;
    public string Kind = "other";
}

/// <summary>What a reader saw on a sheet, for the checking overlay</summary>
public sealed class SheetDebug
{
    public int? Page;                          // 0-based page in the PDF the overlay is drawn on
    public PRect? Legend;
    public List<List<Symbol>> Unknown = new();
    public List<ViewInfo> Views = new();
    public byte[]? Pdf;                        // a picture: the traced drawing the readings are drawn on
}

/// <summary>One drawing sheet (DXF model space / one PDF page), in its own coordinates</summary>
public sealed class Sheet
{
    public string DefaultFloor;
    public List<Room> Rooms = new();
    public List<PointItem> Points = new();
    public List<(string Name, double X, double Y)> FloorTitles = new();
    public double ConfidencePenalty;           // e.g. PDF symbol matching is less certain than DXF blocks
    public double Snap;                        // a point this close outside a room (drawn on its wall) is in that room
    public SheetDebug Debug = new();
    // several drawings on one sheet: (floor / section name, box) of each floor plan or area plan
    public List<(string Name, PRect Box)> Views = new();

    public Sheet(string defaultFloor) => DefaultFloor = defaultFloor;
}
