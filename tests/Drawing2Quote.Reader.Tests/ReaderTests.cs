using System.Text.Json;
using System.Text.Json.Nodes;
using Drawing2Quote.Reader;
using OpenCvSharp;
using Xunit;

namespace Drawing2Quote.Reader.Tests;

/// <summary>The Python reader's tests (tests/test_drawing2quote.py, test_pdf_detection.py, test_raster_detection.py)
/// for the .NET reader, and its output compared with the Python reader's for the same drawings.</summary>
public class ReaderTests
{
    private static readonly string Data = Path.Combine(AppContext.BaseDirectory, "data");
    private static byte[] Read(string name) => File.ReadAllBytes(Path.Combine(Data, name));

    private static Dictionary<string, Dictionary<string, int>> Counts(Extraction ex) =>
        ex.Floors.SelectMany(f => f.Areas).ToDictionary(a => a.Name, a => a.Items);

    private static Dictionary<string, int> D(params (string, int)[] items) => items.ToDictionary(i => i.Item1, i => i.Item2);

    // tests/fixtures.py EXPECTED / UNASSIGNED
    private static readonly Dictionary<string, Dictionary<string, int>> Expected = new()
    {
        ["MASTER BED ROOM"] = D(("light", 6), ("fan", 1), ("socket_5A", 4), ("socket_15A", 1), ("ac_point", 1), ("curtain", 2)),
        ["LOUNGE"] = D(("light", 8), ("fan", 2), ("socket_5A", 6), ("ac_point", 1)),
        ["KITCHEN"] = D(("light", 4), ("exhaust_fan", 1), ("socket_5A", 3), ("socket_15A", 2)),
        ["BED ROOM"] = D(("light", 3), ("fan", 1), ("ac_point", 1)),
        ["GUEST ROOM"] = D(("light", 2), ("fan", 1)),
    };

    // tests/cad_fixture.py EXPECTED
    private static readonly Dictionary<string, Dictionary<string, int>> CadExpected = new()
    {
        ["MASTER BED ROOM"] = D(("ac_point", 1), ("curtain", 1), ("fan", 1), ("light", 4), ("socket_5A", 3), ("switch", 1), ("tv_point", 1)),
        ["LOUNGE"] = D(("ac_point", 1), ("fan", 2), ("light", 6), ("socket_15A", 1), ("socket_5A", 4), ("switch", 2), ("tv_point", 1)),
        ["KITCHEN"] = D(("exhaust_fan", 1), ("light", 2), ("socket_15A", 2), ("socket_5A", 2), ("switch", 1)),
        ["BATH"] = D(("exhaust_fan", 1), ("light", 1), ("switch", 1)),
        ["PASSAGE"] = D(("light", 3), ("switch", 1)),
        ["BED ROOM"] = D(("ac_point", 1), ("fan", 1), ("light", 3), ("socket_5A", 2), ("switch", 1)),
        ["GUEST ROOM"] = D(("fan", 1), ("light", 2), ("socket_5A", 2), ("switch", 1)),
        ["DINING"] = D(("fan", 1), ("light", 3), ("socket_5A", 2)),
    };

    private static void SameItems(Dictionary<string, int> expected, Dictionary<string, int> got, string room) =>
        Assert.True(expected.Count == got.Count && expected.All(kv => got.TryGetValue(kv.Key, out var v) && v == kv.Value),
                    $"{room}: expected {JsonSerializer.Serialize(expected)}, got {JsonSerializer.Serialize(got)}");

    // ── upload rules ──
    [Theory]
    [InlineData("plan.dwg", "AC1027", "Save As -> DXF")]
    [InlineData("photo.jpg", "\xff\xd8\xff\xe0", "damaged or not a picture")]
    [InlineData("plan.heic", "", "HEIC photos")]
    [InlineData("plan.docx", "PK\x03\x04", "not supported")]
    [InlineData("plan.dxf", "hello", "not a valid DXF")]
    public void RejectedFiles(string name, string head, string words)
    {
        var data = head.Select(c => (byte)c).Concat(name.EndsWith(".docx") || name.EndsWith(".dxf") ? Array.Empty<byte>() : new byte[100]).ToArray();
        var e = Assert.Throws<RejectedFileException>(() => DrawingReader.Extract(name, data));
        Assert.Contains(words, e.Message);
    }

    [Fact]
    public void EmptyFileRejected() =>
        Assert.Contains("empty", Assert.Throws<RejectedFileException>(() => DrawingReader.Extract("empty.pdf", Array.Empty<byte>())).Message);

    [Fact]
    public void TinyPictureRejected()
    {
        using var img = new Mat(90, 120, MatType.CV_8UC3, Scalar.All(255));
        var png = img.ImEncode(".png");
        Assert.Contains("too small", Assert.Throws<RejectedFileException>(() => DrawingReader.Extract("plan.png", png)).Message);
    }

    [Fact]
    public void PictureAccepted()
    {
        using var img = new Mat(700, 900, MatType.CV_8UC3, Scalar.All(255));
        Assert.Equal("image", Validator.Check("plan.png", img.ImEncode(".png")));
    }

    // ── reading ──
    [Fact]
    public void DxfExactCounts()
    {
        var ex = DrawingReader.Extract("sample.dxf", Read("sample.dxf"));
        var got = Counts(ex);
        foreach (var (room, items) in Expected) SameItems(items, got[room], room);
        SameItems(D(("light", 1)), got["Unassigned"], "Unassigned");
        Assert.Equal(new[] { "Ground Floor", "First Floor" }, ex.Floors.Select(f => f.Name));
        Assert.Equal("Mr. Ahmed Khan", ex.Project["client"]);
        Assert.Equal("10 Marla", ex.Project["plotSize"]);
        var master = ex.Floors[0].Areas.First(a => a.Name == "MASTER BED ROOM");
        Assert.Equal(196, master.AreaSqft);
        Assert.Equal("14'x14'", master.Dimensions);
        Assert.Equal(1.0, master.Confidence);
        var bed = ex.Floors[1].Areas.First(a => a.Name == "BED ROOM");
        Assert.Equal(168, bed.AreaSqft);
        Assert.Null(bed.Dimensions);                            // from the polyline
        Assert.Contains(ex.Warnings, w => w.Contains("XYZ1"));
        Assert.DoesNotContain(ex.Warnings, w => w.Contains("SOFA"));
        Assert.Equal(6, ex.Legend.Count);
    }

    [Fact]
    public void PdfExactCounts()
    {
        var ex = DrawingReader.Extract("sample.pdf", Read("sample.pdf"));
        var got = Counts(ex);
        foreach (var (room, items) in Expected) SameItems(items, got[room], room);
        SameItems(D(("light", 1)), got["Unassigned"], "Unassigned");
        Assert.Equal(168, ex.Floors[1].Areas.First(a => a.Name == "BED ROOM").AreaSqft);     // rectangle and SCALE 1:72
    }

    [Fact]
    public void CadPdfExactCounts()
    {
        var ex = DrawingReader.Extract("plan.pdf", Read("sample_cad.pdf"));
        var got = Counts(ex);
        foreach (var (room, items) in CadExpected) SameItems(items, got[room], room);
        SameItems(D(("light", 1)), got["Unassigned"], "Unassigned");
        Assert.Equal(CadExpected.Keys.Append("Unassigned").OrderBy(x => x), got.Keys.OrderBy(x => x));
        Assert.Contains(ex.Log, l => l.Contains("8 room(s) (8 traced from walls)"));
        var areas = ex.Floors[0].Areas.ToDictionary(a => a.Name, a => a.AreaSqft);
        Assert.Equal(224, areas["MASTER BED ROOM"]);                           // written 14'x16'
        Assert.InRange(areas["KITCHEN"]!.Value, (12 - 0.75) * (10 - 0.75) * 0.97, (12 - 0.75) * (10 - 0.75) * 1.03);
        Assert.InRange(areas["PASSAGE"]!.Value, (40 - 0.75) * (5 - 0.75) * 0.97, (40 - 0.75) * (5 - 0.75) * 1.03);
        Assert.True(ex.Electrical);
    }

    [Fact]
    public void CadPdfWithoutScale()
    {
        var ex = DrawingReader.Extract("plan.pdf", Read("cad_noscale.pdf"));
        var got = Counts(ex);
        foreach (var (room, items) in CadExpected) SameItems(items, got[room], room);
        var kitchen = ex.Floors[0].Areas.First(a => a.Name == "KITCHEN").AreaSqft!.Value;
        Assert.InRange(kitchen, 11.25 * 9.25 * 0.85, 11.25 * 9.25 * 1.15);
        Assert.Contains(ex.Log, l => l.Contains("no scale written"));
    }

    [Fact]
    public void CadPdfWithoutLegend()
    {
        var ex = DrawingReader.Extract("plan.pdf", Read("cad_nolegend.pdf"));
        Assert.Equal(0, ex.Totals.Points);
        Assert.Equal(CadExpected.Keys.OrderBy(x => x), ex.Floors[0].Areas.Select(a => a.Name).OrderBy(x => x));
        Assert.Contains(ex.Warnings, w => w.Contains("repeated symbol shape"));
        Assert.False(ex.Electrical);
        Assert.StartsWith("Not an electrical drawing", ex.Warnings[0]);
    }

    [Fact]
    public void OverlayMarksRooms()
    {
        var data = Read("sample_cad.pdf");
        var (_, sheets) = DrawingReader.ExtractWithSheets("plan.pdf", data);
        var marked = Overlay.Draw(data, sheets);
        using var doc = new PdfDoc(marked);
        Assert.Equal(1, doc.PageCount);
        var text = doc[0].PlainText();
        Assert.Contains("LOUNGE", text);
        Assert.Contains("224 sq ft", text);
    }

    [Fact]
    public void TextRules()
    {
        Assert.Equal(175.0, Common.ParseDimensions("12'-6\" x 14'-0\"")!.Value.Sqft);
        Assert.Equal(168, Common.ParseDimensions("12 x 14")!.Value.Sqft);
        Assert.Equal(162.8, Common.ParseDimensions("3.6m x 4.2m")!.Value.Sqft);      // (Python: 162.8)
        Assert.Null(Common.ParseDimensions("LOUNGE"));
        Assert.Equal("bedroom", Common.RoomType("MASTER BED ROOM"));
        Assert.Equal("bath", Common.RoomType("Wash Room"));
        Assert.Equal("lounge", Common.RoomType("T.V. LOUNGE"));
        var m = new SymbolMapper();
        Assert.Equal("socket_15A", m.Classify("SKT15A"));
        Assert.Equal("socket_5A", m.Classify("SKT5A"));
        Assert.Equal("exhaust_fan", m.Classify("EXH-FAN"));
        Assert.Equal("", m.Classify("SOFA_3"));
        Assert.Null(m.Classify("XYZ1"));
    }

    // ── symbol shapes ──
    private static Stroke S(IEnumerable<(double, double)> points, bool curve = false)
    {
        var pts = points.Select(p => new Pt(p.Item1, p.Item2)).ToArray();
        return new Stroke(pts, 0.5, false, false, curve, Stroke.BoxOf(pts));
    }

    private static List<Stroke> Light(double cx, double cy, double turn = 0.0)
    {
        var t = PdfGeometry.Linspace(0, 2 * Math.PI, 33);
        var circle = S(t.Select(a => (cx + 4 * Math.Cos(a), cy + 4 * Math.Sin(a))), curve: true);
        var cross = new[] { turn + Math.PI / 4, turn + 3 * Math.PI / 4 }
            .Select(a => S(new[] { (cx - 4 * Math.Cos(a), cy - 4 * Math.Sin(a)), (cx + 4 * Math.Cos(a), cy + 4 * Math.Sin(a)) }));
        return new[] { circle }.Concat(cross).ToList();
    }

    [Fact]
    public void ShapeIgnoresRotationAndPosition()
    {
        var a = PdfGeometry.Describe(Light(0, 0))!;
        var b = PdfGeometry.Describe(Light(100, 50, turn: 0.3))!;
        var plain = PdfGeometry.Describe(Light(0, 0).Take(1).ToList())!;
        Assert.True(PdfGeometry.ShapeDistance(a, b) < 0.15);
        Assert.True(PdfGeometry.ShapeDistance(a, plain) > PdfGeometry.MatchLimit);
    }

    [Fact]
    public void SymbolsGroupedFromSeparateStrokes()
    {
        var items = Light(0, 0).Concat(Light(40, 0)).Append(S(new[] { (-100.0, 20.0), (100.0, 20.0) })).ToList();
        var found = PdfGeometry.FindSymbols(items, 12, 1);
        Assert.Equal(2, found.Count);
        Assert.Equal(new[] { 0L, 40L }, found.Select(s => (long)Math.Round(s.X)).OrderBy(x => x));
    }

    [Fact]
    public void SameFloorOnSeveralSheetsIsMerged()
    {
        Sheet Sheet((string, double, double)[] points) => new("Ground Floor")
        {
            Rooms = new()
            {
                new Room("BATH", Geo.Box(0, 0, 10, 10), 5, 5, "bath", null, 100.0),
                new Room("BATH", Geo.Box(20, 0, 30, 10), 25, 5, "bath", null, 100.0),
                new Room("LOUNGE", Geo.Box(40, 0, 60, 20), 50, 10, "lounge", null, 400.0),
            },
            Points = points.Select(p => new PointItem(p.Item1, p.Item2, p.Item3)).ToList(),
        };
        var lighting = Sheet(new[] { ("light", 2.0, 2.0), ("light", 3.0, 3.0), ("light", 22.0, 2.0), ("light", 45.0, 5.0) });
        var power = Sheet(new[] { ("socket_5A", 2.0, 2.0), ("light", 3.0, 3.0), ("socket_5A", 22.0, 2.0), ("socket_5A", 23.0, 2.0) });
        var floors = Common.Assemble(new[] { lighting, power }, new(), new());
        Assert.Single(floors);
        Assert.Equal(3, floors[0].Areas.Count);                    // two baths stay two, nothing doubled
        var areas = floors[0].Areas.ToDictionary(a => a.Name, a => a.Items);
        SameItems(D(("light", 2), ("socket_5A", 1)), areas["BATH 1"], "BATH 1");   // the light on both sheets counts once
        SameItems(D(("light", 1), ("socket_5A", 2)), areas["BATH 2"], "BATH 2");
        SameItems(D(("light", 1)), areas["LOUNGE"], "LOUNGE");
    }

    // ── pictures ──
    [Fact]
    public void TouchingDotsAreSplit()
    {
        using var img = new Mat(80, 160, MatType.CV_8UC1, Scalar.All(0));
        foreach (var x in new[] { 30, 54, 78, 102 }) Cv2.Circle(img, new Point(x, 40), 12, Scalar.All(1), -1);
        using var outD = RasterReader.SplitDots(img, 200);
        using var lab = new Mat();
        Assert.Equal(4, Cv2.ConnectedComponents(outD, lab, PixelConnectivity.Connectivity8) - 1);
        using var wall = new Mat(80, 300, MatType.CV_8UC1, Scalar.All(0));
        wall[new Rect(10, 30, 280, 20)].SetTo(Scalar.All(1));      // a wall strip is not cut into "dots"
        using var outW = RasterReader.SplitDots(wall, 400);
        Assert.Equal(1, Cv2.ConnectedComponents(outW, lab, PixelConnectivity.Connectivity8) - 1);
    }

    [Fact]
    public void ThinningAndTracing()
    {
        using var img = new Mat(120, 200, MatType.CV_8UC1, Scalar.All(0));
        Cv2.Circle(img, new Point(60, 60), 30, Scalar.All(1), 5);
        Cv2.Line(img, new Point(100, 20), new Point(190, 100), Scalar.All(1), 4);
        var sk = RasterReader.Thin(img, out int w, out int h);
        // one pixel wide: no 2x2 block of ink left
        for (int y = 0; y + 1 < h; y++)
            for (int x = 0; x + 1 < w; x++)
                Assert.False(sk[y * w + x] != 0 && sk[y * w + x + 1] != 0 && sk[(y + 1) * w + x] != 0 && sk[(y + 1) * w + x + 1] != 0);
        var lines = RasterReader.Trace(sk, w, h, 4);
        Assert.Contains(lines, l => Math.Abs(l.Pts[0].X - l.Pts[^1].X) < 1e-9 && Math.Abs(l.Pts[0].Y - l.Pts[^1].Y) < 1e-9);   // the circle comes back as a ring
    }

    [Fact]
    public void RoomsWithoutReadableNamesComeFromTheWalls()
    {
        using var walls = new Mat(600, 900, MatType.CV_8UC1, Scalar.All(0));
        Cv2.Rectangle(walls, new Point(50, 50), new Point(850, 550), Scalar.All(1), 14);      // outer walls
        Cv2.Line(walls, new Point(450, 50), new Point(450, 550), Scalar.All(1), 14);           // one wall across, with a door gap
        walls[new Rect(443, 280, 15, 50)].SetTo(Scalar.All(0));
        Cv2.Line(walls, new Point(450, 300), new Point(850, 300), Scalar.All(1), 14);
        var found = new List<double>();
        var outlines = new List<List<Pt>>();
        var lines = RasterReader.UnnamedRooms(walls, (20, 20, 880, 580), new(), 0.24, roomsPx: found, outlines: outlines);
        Assert.Equal(new[] { "ROOM 1", "ROOM 2", "ROOM 3" }, lines.Select(l => l.T.Text));
        Assert.Equal(3, outlines.Count);
        var (n, how) = RasterReader.PictureScale(new(), (20, 20, 880, 580), found, 0.24);
        Assert.NotNull(n);
        Assert.Contains("estimated", how);
    }

    [Fact]
    public void ScannedPlanSymbols()
    {
        if (!Ocr.Available()) return;
        var ex = DrawingReader.Extract("plan.jpg", Read("scanned.jpg"));
        Assert.Equal("image", ex.FileType);
        Assert.Contains(ex.Warnings, w => w.Contains("read from a picture"));
        var got = ex.Totals.Items;
        Assert.Equal(25, got.GetValueOrDefault("light"));
    }

    // ── Python helpers ──
    [Fact]
    public void PythonRoundingAndFormatting()
    {
        Assert.Equal(2.67, Py.Round(2.675, 2));            // the binary value is just under 2.675
        Assert.Equal(0.12, Py.Round(0.125, 2));            // an exact tie: to even
        Assert.Equal(2.0, Py.Round(2.5));
        Assert.Equal("1.0", Py.Repr(1.0));
        Assert.Equal("0.65", Py.Repr(0.65));
        Assert.Equal("1e-05", Py.Repr(0.00001));
        Assert.Equal("1e+16", Py.Repr(1e16));
        Assert.Equal("224", Py.G(224.0));
        Assert.Equal("12.5", Py.G(12.5));
        Assert.Equal("1.12", Py.FixedFast(1.125, 2));
        Assert.Equal("-0.00", Py.FixedFast(-0.001, 2));
        Assert.Equal("2", Py.Fixed(2.5, 0));
        Assert.Equal(5 / 6.0, Py.SequenceRatio("LIVIRG", "LIVING"), 12);     // difflib: 0.8333333333333334
        Assert.Equal(-1.0 + Math.PI, Py.Mod(-1.0, Math.PI), 12);
    }

    // ── the same output as the Python reader ──
    [Theory]
    [InlineData("sample.dxf")]
    [InlineData("sample.pdf")]
    [InlineData("sample_cad.pdf")]
    [InlineData("cad_noscale.pdf")]
    [InlineData("cad_nolegend.pdf")]
    [InlineData("scanned.jpg")]
    public void SameAsPythonReader(string name)
    {
        if (name.EndsWith(".jpg") && !Ocr.Available()) return;
        var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(Data, $"python_{name}.json")))!;
        var got = JsonNode.Parse(DrawingReader.Extract(name, Read(name)).ToJson())!;
        var diffs = new List<string>();
        Compare(expected, got, "", diffs);
        Assert.True(diffs.Count == 0, string.Join("\n", diffs.Take(20)));
    }

    private static void Compare(JsonNode? a, JsonNode? b, string path, List<string> diffs)
    {
        switch (a)
        {
            case JsonObject oa when b is JsonObject ob:
                if (!oa.Select(kv => kv.Key).SequenceEqual(ob.Select(kv => kv.Key))) diffs.Add($"{path}: keys differ");
                foreach (var (k, v) in oa) if (ob.ContainsKey(k)) Compare(v, ob[k], $"{path}.{k}", diffs);
                break;
            case JsonArray la when b is JsonArray lb:
                if (la.Count != lb.Count) diffs.Add($"{path}: {la.Count} items != {lb.Count}");
                for (int i = 0; i < Math.Min(la.Count, lb.Count); i++) Compare(la[i], lb[i], $"{path}[{i}]", diffs);
                break;
            case JsonValue va when b is JsonValue vb && va.TryGetValue<double>(out var x) && vb.TryGetValue<double>(out var y):
                if (Math.Abs(x - y) > 1e-9 * Math.Max(1, Math.Abs(x))) diffs.Add($"{path}: {x} != {y}");
                break;
            default:
                if (a?.ToJsonString() != b?.ToJsonString()) diffs.Add($"{path}: {a?.ToJsonString()} != {b?.ToJsonString()}");
                break;
        }
    }
}
