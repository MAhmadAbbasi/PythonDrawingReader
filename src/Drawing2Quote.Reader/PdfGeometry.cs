using System.Diagnostics;
using System.Text.RegularExpressions;
using NetTopologySuite.Geometries;
using NetTopologySuite.Operation.Buffer;

namespace Drawing2Quote.Reader;

/// <summary>A drawn path as PyMuPDF's page.get_drawings() gives it</summary>
public sealed class Drawing
{
    public string Type = "s";             // "s" stroke, "f" fill, "fs" both
    public double? Width;
    public double[]? Fill;                // RGB, null when not filled
    public double[]? Color;               // RGB stroke colour
    public string? Dashes;
    public double? FillOpacity;
    public PRect Rect;
    public List<DrawItem> Items = new();
}

/// <summary>A piece of a path: "l" (P0, P1), "c" (P0..P3), "re" (P0 = top left, P1 = bottom right), "qu" (ul, ur, ll, lr)</summary>
public sealed class DrawItem
{
    public string Op;
    public Pt[] P;

    public DrawItem(string op, params Pt[] p)
    {
        Op = op; P = p;
    }
}

/// <summary>A drawn path broken into one simple piece</summary>
public sealed class Stroke
{
    public Pt[] Pts;                       // polyline
    public double Width;                   // line weight; filled shapes rank as the heaviest
    public bool Filled, Dashed, Curve;
    public (double X0, double Y0, double X1, double Y1) BBox;
    public string Color = "";              // "r,g,b" in tenths (fill colour for filled shapes)
    public double PathSize;                // size of the drawn path this stroke is part of
    public double ChainSize;               // size of the run of strokes joined end to end it belongs to
    public int PathId = -1;                // which drawn path it came from
    internal int Order;                    // creation order (Python's id() order for ties)

    public Stroke(Pt[] pts, double width, bool filled, bool dashed, bool curve, (double, double, double, double) bbox,
                  string color = "", double pathSize = 0.0, int pathId = -1)
    {
        Pts = pts; Width = width; Filled = filled; Dashed = dashed; Curve = curve; BBox = bbox; Color = color;
        PathSize = pathSize; PathId = pathId;
    }

    /// <summary>a curve flattened into tiny lines (as printer drivers do) counts as the whole curve</summary>
    public double PieceSize => Math.Max(Size, ChainSize);

    public double Size => Math.Max(BBox.X1 - BBox.X0, BBox.Y1 - BBox.Y0);

    public bool Inside((double X0, double Y0, double X1, double Y1) r) =>
        r.X0 <= BBox.X0 && r.Y0 <= BBox.Y0 && BBox.X1 <= r.X1 && BBox.Y1 <= r.Y1;

    public static (double, double, double, double) BoxOf(Pt[] pts)
    {
        double x0 = double.PositiveInfinity, y0 = double.PositiveInfinity, x1 = double.NegativeInfinity, y1 = double.NegativeInfinity;
        foreach (var p in pts)
        {
            x0 = Math.Min(x0, p.X); y0 = Math.Min(y0, p.Y);
            x1 = Math.Max(x1, p.X); y1 = Math.Max(y1, p.Y);
        }
        return (x0, y0, x1, y1);
    }
}

/// <summary>A group of small strokes that touch: one symbol, with a shape description that ignores rotation and
/// position</summary>
public sealed class Symbol
{
    public (double X0, double Y0, double X1, double Y1) BBox;
    public double X, Y;
    public double R;                       // furthest ink from the centre
    public double Length;                  // total ink length / r
    public double[] Radial = new double[6];// how the ink is spread from the centre outwards (6 bins)
    public double[] Dirs = new double[12]; // line directions (12 bins over 180 degrees)
    public double Curve;                   // share of the ink that is curved
    public bool Filled;
    public double W, H;
    public string Text = "";               // a short text inside the symbol ("S", "TV" ...)
    public string Color = "";              // the colour most of the ink has
    public double FillArea;                // area covered by filled parts (dots), in pt²
    internal double[][]? Rolls;

    public double Size => Math.Max(W, H);
}

/// <summary>
/// Geometry of a vector PDF page, used by the PDF reader:
/// Strokes() - every drawn path broken into simple strokes (lines, curves as short lines, rectangles);
/// FindSymbols() - small strokes that touch are grouped into one symbol; each symbol gets a shape description that
/// ignores rotation and position, so a socket turned 90 degrees on a wall still matches the legend symbol;
/// RoomFinder - room outlines from the wall lines: door openings are closed by extending wall ends across the gap,
/// and the enclosed space around each room name is the room.
/// </summary>
public static class PdfGeometry
{
    private static readonly Regex Dashed = new(@"\[\s*[\d.]*[1-9]", Py.RX);   // "[3 2] 0" is dashed, "[] 0" is solid
    public const int MaxWallSegments = 30000;      // a line set bigger than this is not tried for rooms (too slow)
    public static double RoomTimeLimit = 45.0;     // seconds of room tracing per page, then the rest are names only
    public const double MatchLimit = 0.5;          // shape distance under which a symbol is the legend symbol
    public const double SameShape = 0.3;           // two unknown symbols with the same shape

    // ── strokes ──
    private static string Colour(double[]? rgb) =>
        rgb is { Length: > 0 } ? string.Join(",", rgb.Take(3).Select(c => Py.RoundInt(c * 10).ToString(Py.Inv))) : "";

    private static readonly double[] BezierT = Linspace(0, 1, 8);

    /// <summary>numpy.linspace(start, stop, num)</summary>
    internal static double[] Linspace(double start, double stop, int num)
    {
        var y = new double[num];
        int div = num - 1;
        double step = (stop - start) / div;
        for (int i = 0; i < num; i++) y[i] = i * step + start;
        if (num > 1) y[num - 1] = stop;
        return y;
    }

    private static Pt[] Bezier(Pt p0, Pt p1, Pt p2, Pt p3)
    {
        var out_ = new Pt[BezierT.Length];
        for (int i = 0; i < BezierT.Length; i++)
        {
            double t = BezierT[i], u = 1 - t;
            double a = Math.Pow(u, 3), b = 3 * (u * u) * t, c = 3 * u * (t * t), d = Math.Pow(t, 3);
            out_[i] = new Pt(a * p0.X + b * p1.X + c * p2.X + d * p3.X, a * p0.Y + b * p1.Y + c * p2.Y + d * p3.Y);
        }
        return out_;
    }

    /// <summary>matrix: turns PyMuPDF's unrotated coordinates into the page as shown (page.rotation_matrix)</summary>
    public static List<Stroke> Strokes(IList<Drawing> drawings, Mat2D? matrix = null)
    {
        var out_ = new List<Stroke>();
        double heaviest = (drawings.Count > 0 ? drawings.Max(d => d.Width ?? 0.0) : 0.0) + 1.0;
        for (int pathId = 0; pathId < drawings.Count; pathId++)
        {
            var d = drawings[pathId];
            var fill = d.Fill is { Length: > 0 } ? d.Fill : null;
            double fo = d.FillOpacity is null or 0 ? 1.0 : d.FillOpacity.Value;
            bool filled = fill != null && fill.Min() < 0.9 && fo > 0.2;
            bool stroked = (string.IsNullOrEmpty(d.Type) ? "s" : d.Type).Contains('s');
            bool dashed = stroked && Dashed.IsMatch(d.Dashes ?? "");
            double width = filled ? heaviest : (d.Width ?? 0.0);
            var color = Colour(filled ? fill : d.Color);
            double pathSize = Math.Max(d.Rect.Width, d.Rect.Height);
            int first = out_.Count;
            var chains = new List<int>();          // chain number of each stroke of this path
            Pt? lastEnd = null;
            foreach (var item in d.Items)
            {
                Pt[] pts;
                bool curve = false;
                switch (item.Op)
                {
                    case "l": pts = new[] { item.P[0], item.P[1] }; break;
                    case "c": pts = Bezier(item.P[0], item.P[1], item.P[2], item.P[3]); curve = true; break;
                    case "re":
                        {
                            double x0 = item.P[0].X, y0 = item.P[0].Y, x1 = item.P[1].X, y1 = item.P[1].Y;
                            pts = new[] { new Pt(x0, y0), new Pt(x1, y0), new Pt(x1, y1), new Pt(x0, y1), new Pt(x0, y0) };
                            break;
                        }
                    case "qu":
                        {
                            Pt ul = item.P[0], ur = item.P[1], ll = item.P[2], lr = item.P[3];
                            pts = new[] { ul, ur, lr, ll, ul };
                            break;
                        }
                    default: continue;
                }
                if (matrix is { } m)
                    pts = pts.Select(p => new Pt(p.X * m.A + p.Y * m.C + m.E, p.X * m.B + p.Y * m.D + m.F)).ToArray();
                bool line = item.Op is "l" or "c";
                bool joined = lastEnd is { } le && line && Math.Max(Math.Abs(pts[0].X - le.X), Math.Abs(pts[0].Y - le.Y)) < 0.01;
                chains.Add(joined && chains.Count > 0 ? chains[^1] : (chains.Count > 0 ? chains[^1] + 1 : 0));
                lastEnd = line ? pts[^1] : null;
                out_.Add(new Stroke(pts, width, filled, dashed, curve, Stroke.BoxOf(pts), color, pathSize, pathId) { Order = out_.Count });
            }
            if (chains.Count > 0)
            {
                var boxOf = new Dictionary<int, double[]>();
                for (int k = 0; k < chains.Count; k++)
                {
                    var m = out_[first + k];
                    if (!boxOf.TryGetValue(chains[k], out var bb)) boxOf[chains[k]] = bb = new[] { m.BBox.X0, m.BBox.Y0, m.BBox.X1, m.BBox.Y1 };
                    bb[0] = Math.Min(bb[0], m.BBox.X0); bb[1] = Math.Min(bb[1], m.BBox.Y0);
                    bb[2] = Math.Max(bb[2], m.BBox.X1); bb[3] = Math.Max(bb[3], m.BBox.Y1);
                }
                for (int k = 0; k < chains.Count; k++)
                {
                    var bb = boxOf[chains[k]];
                    out_[first + k].ChainSize = Math.Max(bb[2] - bb[0], bb[3] - bb[1]);
                }
            }
        }
        return out_;
    }

    /// <summary>start points, end points, width, is-curve, is-filled of every straight piece</summary>
    internal static (Pt[] P0, Pt[] P1, double[] W, bool[] C, bool[] F) Segments(IEnumerable<Stroke> items)
    {
        var p0 = new List<Pt>(); var p1 = new List<Pt>(); var w = new List<double>(); var c = new List<bool>(); var f = new List<bool>();
        foreach (var s in items)
        {
            int n = s.Pts.Length - 1;
            if (n < 1) continue;
            for (int i = 0; i < n; i++)
            {
                p0.Add(s.Pts[i]); p1.Add(s.Pts[i + 1]); w.Add(s.Width); c.Add(s.Curve); f.Add(s.Filled);
            }
        }
        return (p0.ToArray(), p1.ToArray(), w.ToArray(), c.ToArray(), f.ToArray());
    }

    // ── symbols ──
    private static double NpRound2(double v)
    {
        var r = Math.Round(v * 100.0, MidpointRounding.ToEven) / 100.0;
        return r == 0 ? 0.0 : r;
    }

    private static double Norm(double dx, double dy) => Math.Sqrt(dx * dx + dy * dy);

    public static Symbol? Describe(IList<Stroke> items)
    {
        var (P0, P1, _, C, F) = Segments(items);
        // the same line drawn twice (common in CAD exports) counts once
        var seen = new HashSet<(double, double, double, double)>();
        var keep = new List<int>();
        for (int i = 0; i < P0.Length; i++)
        {
            double ax = NpRound2(P0[i].X), ay = NpRound2(P0[i].Y), bx = NpRound2(P1[i].X), by = NpRound2(P1[i].Y);
            bool swap = ax > bx || (ax == bx && ay > by);
            var key = swap ? (bx, by, ax, ay) : (ax, ay, bx, by);
            if (seen.Add(key)) keep.Add(i);
        }
        int n = keep.Count;
        var p0 = keep.Select(i => P0[i]).ToArray();
        var p1 = keep.Select(i => P1[i]).ToArray();
        var curve = keep.Select(i => C[i]).ToArray();
        var filled = keep.Select(i => F[i]).ToArray();
        var lengths = new double[n];
        for (int i = 0; i < n; i++) lengths[i] = Norm(p1[i].X - p0[i].X, p1[i].Y - p0[i].Y);
        double total = Py.NpSum(lengths);
        if (total <= 0) return null;
        double lox = double.PositiveInfinity, loy = double.PositiveInfinity, hix = double.NegativeInfinity, hiy = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            lox = Math.Min(lox, Math.Min(p0[i].X, p1[i].X)); loy = Math.Min(loy, Math.Min(p0[i].Y, p1[i].Y));
            hix = Math.Max(hix, Math.Max(p0[i].X, p1[i].X)); hiy = Math.Max(hiy, Math.Max(p0[i].Y, p1[i].Y));
        }
        double w = hix - lox, h = hiy - loy;
        // sample the ink evenly along its length
        double step = Math.Max(w, h) / 24;
        if (step == 0) step = 1.0;
        var ks = new long[n];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            ks[i] = (long)Math.Max(1, Math.Ceiling(lengths[i] / step));
            count += (int)ks[i];
        }
        var sx = new double[count]; var sy = new double[count]; var wts = new double[count];
        int s = 0;
        for (int i = 0; i < n; i++)
            for (long j = 0; j < ks[i]; j++, s++)
            {
                double t = (j + 0.5) / ks[i];
                sx[s] = p0[i].X + (p1[i].X - p0[i].X) * t;
                sy[s] = p0[i].Y + (p1[i].Y - p0[i].Y) * t;
                wts[s] = lengths[i] / ks[i];
            }
        double cx = 0, cy = 0;
        for (int k = 0; k < count; k++) { cx += wts[k] * sx[k]; cy += wts[k] * sy[k]; }
        double wsum = Py.NpSum(wts);
        cx /= wsum; cy /= wsum;
        double r = 0;
        for (int i = 0; i < n; i++)
        {
            r = Math.Max(r, Norm(p0[i].X - cx, p0[i].Y - cy));
        }
        for (int i = 0; i < n; i++) r = Math.Max(r, Norm(p1[i].X - cx, p1[i].Y - cy));
        r = Math.Max(r, 1e-6);
        // the spread from the centre: numpy.histogram(..., bins=6, range=(0, 1.0001), weights)
        var radial = new double[6];
        var edges = Linspace(0, 1.0001, 7);
        double first = 0, last = 1.0001;
        for (int b0 = 0; b0 < count; b0 += 65536)
        {
            var block = new double[6];
            int end = Math.Min(count, b0 + 65536);
            for (int k = b0; k < end; k++)
            {
                double v = Norm(sx[k] - cx, sy[k] - cy) / r;
                if (!(v >= first && v <= last)) continue;
                int idx = (int)((v - first) / (last - first) * 6);
                if (idx == 6) idx -= 1;
                if (v < edges[idx]) idx -= 1;
                if (v >= edges[idx + 1] && idx != 5) idx += 1;
                block[idx] += wts[k];
            }
            for (int q = 0; q < 6; q++) radial[q] += block[q];
        }
        for (int q = 0; q < 6; q++) radial[q] /= total;
        // directions, softly split between the two nearest bins so a 1-degree turn changes little
        var dirs = new double[12];
        var low = new int[n]; var frac = new double[n];
        for (int i = 0; i < n; i++)
        {
            double ang = Py.Mod(Math.Atan2(p1[i].Y - p0[i].Y, p1[i].X - p0[i].X), Math.PI);
            double pos = ang / (Math.PI / 12);
            double fl = Math.Floor(pos);
            low[i] = (int)(((long)fl % 12 + 12) % 12);
            frac[i] = pos - fl;
        }
        for (int i = 0; i < n; i++) dirs[low[i]] += lengths[i] * (1 - frac[i]);
        for (int i = 0; i < n; i++) dirs[(low[i] + 1) % 12] += lengths[i] * frac[i];
        double fillArea = 0.0;
        var filledItems = items.Where(st => st.Filled).ToList();
        if (filledItems.Count > 0)
        {
            // printer drivers paint a filled dot as many thin slivers: measure the area they cover together
            var shapes = filledItems.Select(st => st.Pts.Length >= 4 ? (Geometry)Geo.Polygon(st.Pts) : Geo.Line(st.Pts));
            fillArea = Geo.BufferUnion(shapes, 0.2, quadSegs: 1).Area;
        }
        var ink = new Dictionary<string, double>();
        foreach (var st in items)
        {
            var seg = new double[Math.Max(0, st.Pts.Length - 1)];
            for (int i = 0; i + 1 < st.Pts.Length; i++) seg[i] = Norm(st.Pts[i + 1].X - st.Pts[i].X, st.Pts[i + 1].Y - st.Pts[i].Y);
            ink[st.Color] = (ink.TryGetValue(st.Color, out var v) ? v : 0.0) + Py.NpSum(seg);
        }
        double curveLen = Py.NpSum(Enumerable.Range(0, n).Where(i => curve[i]).Select(i => lengths[i]).ToArray());
        double filledLen = Py.NpSum(Enumerable.Range(0, n).Where(i => filled[i]).Select(i => lengths[i]).ToArray());
        return new Symbol
        {
            FillArea = fillArea, Color = ink.Count > 0 ? Py.MaxBy(ink, kv => kv.Value).Key : "",
            BBox = (lox, loy, hix, hiy), X = cx, Y = cy, R = r, Length = total / r,
            Radial = radial, Dirs = dirs.Select(v => v / total).ToArray(), Curve = curveLen / total,
            Filled = filledLen / total > 0.5, W = w, H = h,
        };
    }

    /// <summary>the 12 rotations of a symbol's direction histogram (kept on the symbol: legend symbols are compared often)</summary>
    private static double[][] Rolls(Symbol sym)
    {
        if (sym.Rolls == null)
        {
            var r = new double[12][];
            for (int k = 0; k < 12; k++)
            {
                r[k] = new double[12];
                for (int i = 0; i < 12; i++) r[k][i] = sym.Dirs[((i - k) % 12 + 12) % 12];
            }
            sym.Rolls = r;
        }
        return sym.Rolls;
    }

    /// <summary>a solid round dot: filled for a good part of its circle, as wide as high, round (a square's corners
    /// reach further out from its centre)</summary>
    private static bool Disc(Symbol s) =>
        s.Filled && s.R > 0 && 0.8 <= s.W / Math.Max(s.H, 1e-6) && s.W / Math.Max(s.H, 1e-6) <= 1.25
        && s.FillArea >= 0.25 * Math.PI * Math.Pow(s.R, 2) && s.R <= 1.15 * Math.Max(s.W, s.H) / 2;

    /// <summary>0 for the same shape at any rotation; about 0.5 and more for different shapes.
    /// sizeWeight: how much a size difference counts (less for a legend drawn on another sheet)</summary>
    public static double ShapeDistance(Symbol a, Symbol b, double sizeWeight = 0.4)
    {
        if (a.Filled != b.Filled || a.Text != b.Text) return double.PositiveInfinity;
        if (Disc(a) && Disc(b))
            // a solid round dot: drawn as one fill on the plan and as a hatched circle in the legend, the ink differs
            // but the shape is the same
            return sizeWeight * Math.Abs(Math.Log(a.R / b.R)) + Math.Abs(a.FillArea / Math.Pow(a.R, 2) - b.FillArea / Math.Pow(b.R, 2)) / Math.PI;
        // plans often draw symbols at another size than the legend (a fan at its real sweep), so size counts little
        double d = sizeWeight * Math.Abs(Math.Log(a.R / b.R));
        if (a.Color.Length > 0 && b.Color.Length > 0 && a.Color != b.Color) d += 0.4;
        d += Math.Abs(a.Length - b.Length) / Math.Max(a.Length, b.Length);
        double rs = 0;
        for (int i = 0; i < 6; i++) rs += Math.Abs(a.Radial[i] - b.Radial[i]);
        d += 0.5 * rs;
        d += Math.Abs(a.Curve - b.Curve);
        var rolls = Rolls(b);
        double best = double.PositiveInfinity;
        var row = new double[12];
        for (int k = 0; k < 12; k++)
        {
            for (int i = 0; i < 12; i++) row[i] = Math.Abs(rolls[k][i] - a.Dirs[i]);
            best = Math.Min(best, Py.NpSum(row));
        }
        d += 0.5 * best;
        // the outline box, allowing quarter turns (a square and the same square standing on a corner differ)
        double aw = Math.Max(a.W, 0.05 * a.R), ah = Math.Max(a.H, 0.05 * a.R);
        double bw = Math.Max(b.W, 0.05 * b.R), bh = Math.Max(b.H, 0.05 * b.R);
        d += 0.25 * Math.Min(Math.Abs(Math.Log(aw / bw)) + Math.Abs(Math.Log(ah / bh)), Math.Abs(Math.Log(aw / bh)) + Math.Abs(Math.Log(ah / bw)));
        return d;
    }

    /// <summary>For each stroke, the size of the run of strokes joined end to end through plain bends (exactly two
    /// pieces meeting). A wire drawn as hundreds of tiny separate pieces is one long run and is left out; a symbol
    /// touching a wire meets it at a junction (three pieces), so the symbol stays separate.</summary>
    private static List<double> RunExtent(IList<Stroke> items)
    {
        if (items.Count == 0) return new List<double>();
        static (long, long) Vid(Pt p) => ((long)Math.Round(p.X * 20, MidpointRounding.ToEven), (long)Math.Round(p.Y * 20, MidpointRounding.ToEven));   // 0.05 pt grid
        var edgeOf = new Dictionary<((long, long), (long, long)), int>();
        var edges = new List<((long, long) A, (long, long) B)>();
        var strokeEdge = new List<int>();
        foreach (var st in items)
        {
            var a = Vid(st.Pts[0]); var b = Vid(st.Pts[^1]);
            if (a == b) { strokeEdge.Add(-1); continue; }
            var key = a.CompareTo(b) <= 0 ? (a, b) : (b, a);
            if (!edgeOf.TryGetValue(key, out var e)) { edgeOf[key] = e = edges.Count; edges.Add(key); }
            strokeEdge.Add(e);
        }
        var at = new Dictionary<(long, long), List<int>>();
        for (int e = 0; e < edges.Count; e++)
        {
            foreach (var v in new[] { edges[e].A, edges[e].B })
            {
                if (!at.TryGetValue(v, out var l)) at[v] = l = new List<int>();
                l.Add(e);
            }
        }
        var parent = Enumerable.Range(0, edges.Count).ToArray();
        int Root(int i)
        {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        }
        foreach (var es in at.Values)
        {
            if (es.Count == 2)
            {
                int ra = Root(es[0]), rb = Root(es[1]);
                if (ra != rb) parent[ra] = rb;
            }
        }
        var boxOf = new Dictionary<int, double[]>();
        for (int k = 0; k < items.Count; k++)
        {
            int e = strokeEdge[k];
            if (e < 0) continue;
            int r = Root(e);
            var st = items[k];
            if (!boxOf.TryGetValue(r, out var bb)) boxOf[r] = bb = new[] { st.BBox.X0, st.BBox.Y0, st.BBox.X1, st.BBox.Y1 };
            bb[0] = Math.Min(bb[0], st.BBox.X0); bb[1] = Math.Min(bb[1], st.BBox.Y0);
            bb[2] = Math.Max(bb[2], st.BBox.X1); bb[3] = Math.Max(bb[3], st.BBox.Y1);
        }
        var out_ = new List<double>(items.Count);
        for (int k = 0; k < items.Count; k++)
        {
            int e = strokeEdge[k];
            if (e < 0) out_.Add(items[k].Size);
            else
            {
                var bb = boxOf[Root(e)];
                out_.Add(Math.Max(bb[2] - bb[0], bb[3] - bb[1]));
            }
        }
        return out_;
    }

    /// <summary>strokes that are the dashes of a dashed line drawn as separate short lines (CAD linework exploded into
    /// pieces: wiring, ceiling lines): 4 or more short straight pieces of about the same length, on one line, evenly
    /// spaced</summary>
    public static HashSet<Stroke> DrawnDashes(IList<Stroke> items)
    {
        var lines = new Dictionary<(long, long), List<(double A, double B, double Len, int Order, Stroke S)>>();
        foreach (var s in items)
        {
            if (s.Pts.Length != 2 || s.Filled) continue;
            double x0 = s.Pts[0].X, y0 = s.Pts[0].Y, x1 = s.Pts[1].X, y1 = s.Pts[1].Y;
            double length = PyHypot(x1 - x0, y1 - y0);
            if (!(0.3 <= length && length <= 15)) continue;
            double ang = Py.Mod(Math.Atan2(y1 - y0, x1 - x0), Math.PI);
            double ux = Math.Cos(ang), uy = Math.Sin(ang);
            double offset = -uy * x0 + ux * y0;                 // distance of its line from the origin
            double a = ux * x0 + uy * y0, b = ux * x1 + uy * y1; // where it runs along that line
            var key = ((((long)Math.Round(ang * 180 / Math.PI, MidpointRounding.ToEven)) % 180 + 180) % 180,
                       (long)Math.Round(offset / 0.3, MidpointRounding.ToEven));
            if (!lines.TryGetValue(key, out var l)) lines[key] = l = new();
            l.Add((Math.Min(a, b), Math.Max(a, b), length, s.Order, s));
        }
        var out_ = new HashSet<Stroke>();     // (strokes compare by reference)
        foreach (var raw in lines.Values)
        {
            if (raw.Count < 4) continue;
            var pieces = raw.OrderBy(p => p.A).ThenBy(p => p.B).ThenBy(p => p.Len).ThenBy(p => p.Order).ToList();
            var run = new List<(double A, double B, double Len, int Order, Stroke S)> { pieces[0] };
            for (int k = 1; k <= pieces.Count; k++)
            {
                var q = run[^1];
                if (k < pieces.Count)
                {
                    var p = pieces[k];
                    if (0.1 < p.A - q.B && p.A - q.B < 2.5 * Math.Max(p.Len, q.Len) && 0.6 < p.Len / q.Len && p.Len / q.Len < 1.67)
                    {
                        run.Add(p);
                        continue;
                    }
                }
                if (run.Count >= 4)
                {
                    var gaps = Enumerable.Range(0, run.Count - 1).Select(i => run[i + 1].A - run[i].B).ToList();
                    if (gaps.Max() < 2 * gaps.Min() + 0.3)       // evenly spaced
                        foreach (var r in run) out_.Add(r.S);
                }
                if (k < pieces.Count) run = new() { pieces[k] };
            }
        }
        return out_;
    }

    /// <summary>math.hypot (Python 3.11 is accurate to the last bit where a plain square root is not always)</summary>
    internal static double PyHypot(double x, double y) => double.Hypot(x, y);

    /// <summary>connected groups of n items joined by the pairs: each item's label is the smallest index in its group</summary>
    internal static int[] Components(int n, IEnumerable<(int A, int B)> pairs)
    {
        var parent = Enumerable.Range(0, n).ToArray();
        int Root(int i)
        {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        }
        foreach (var (a, b) in pairs)
        {
            if (a == b) continue;
            int ra = Root(a), rb = Root(b);
            if (ra != rb)
            {
                if (ra < rb) parent[rb] = ra; else parent[ra] = rb;
            }
        }
        var label = new int[n];
        for (int i = 0; i < n; i++) label[i] = Root(i);     // the root is the group's smallest index (smaller roots win)
        return label;
    }

    private static string Family(string c)
    {
        // (near-black shades count as one - a dot's fill and outline differ slightly; a pale grey / white halo round a
        // symbol joins whatever it surrounds)
        if (c.Length == 0) return c;
        var v = c.Split(',').Select(x => int.Parse(x, Py.Inv)).ToArray();
        return v.Min() >= 7 ? "pale" : v.Max() <= 3 ? "dark" : c;
    }

    /// <summary>Groups strokes no bigger than maxSize that touch (within gap) into symbols. colours: only strokes in
    /// these colours (the legend symbols' colours - leaves out grey furniture and walls that touch the symbols).
    /// maxLine: a single straight line longer than this is wiring, not part of a symbol.
    /// sameColour: only strokes of one colour join (a red light drawn on a black dimension line)</summary>
    public static List<Symbol> FindSymbols(IList<Stroke> items, double maxSize, double gap, ISet<string>? colours = null,
                                           double? maxLine = null, bool sameColour = false)
    {
        var dashes = DrawnDashes(items);
        var small = items.Where(s => s.PieceSize <= maxSize && !s.Dashed && !dashes.Contains(s)
                                     && (colours == null || colours.Count == 0 || colours.Contains(s.Color))).ToList();
        if (maxLine is { } ml && ml != 0)
            small = small.Where(s => s.Pts.Length > 2 || s.Size <= ml).ToList();
        var ext = RunExtent(small);
        small = small.Where((s, i) => ext[i] <= maxSize).ToList();
        if (small.Count == 0) return new List<Symbol>();
        var b = small.Select(s => s.BBox).ToArray();
        var tree = new Geo.Tree(b.Select(x => Geo.Env(x.X0, x.Y0, x.X1, x.Y1)));
        var pairs = new List<(int A, int B)>();
        for (int i = 0; i < b.Length; i++)
            foreach (int j in tree.Query(Geo.Env(b[i].X0 - gap, b[i].Y0 - gap, b[i].X1 + gap, b[i].Y1 + gap)))
                pairs.Add((i, j));
        if (sameColour && pairs.Count > 0)
        {
            var colour = small.Select(s => Family(s.Color)).ToArray();
            pairs = pairs.Where(p => colour[p.A] == colour[p.B] || colour[p.A] == "pale" || colour[p.B] == "pale").ToList();
        }
        // an outline far bigger than what lies in its box (a room drawn as one rectangle, the fan in the room) joins
        // it only by touching its lines; parts of one symbol drawn one inside the other (a square in a circle) join
        // by their boxes as before
        var isOutline = small.Select(s => s.Pts.Length > 2 && !s.Filled).ToArray();
        if (isOutline.Any(x => x) && pairs.Count > 0)
        {
            var size = b.Select(x => Math.Max(x.X1 - x.X0, x.Y1 - x.Y0)).ToArray();
            var geoms = new Geometry?[small.Count];
            Geometry G(int k) => geoms[k] ??= small[k].Filled && small[k].Pts.Length >= 4 ? Geo.Polygon(small[k].Pts)
                : small[k].Pts.Length >= 2 ? Geo.Line(small[k].Pts) : Geo.Point(small[k].Pts[0].X, small[k].Pts[0].Y);
            pairs = pairs.Where(p =>
            {
                bool check = (isOutline[p.A] && size[p.A] > 4 * size[p.B]) || (isOutline[p.B] && size[p.B] > 4 * size[p.A]);
                return !check || G(p.A).IsWithinDistance(G(p.B), gap);
            }).ToList();
        }
        var label = Components(small.Count, pairs);
        var groups = new Dictionary<int, List<int>>();
        var order = new List<int>();
        for (int i = 0; i < small.Count; i++)
        {
            if (!groups.TryGetValue(label[i], out var g)) { groups[label[i]] = g = new List<int>(); order.Add(label[i]); }
            g.Add(i);
        }
        var out_ = new List<Symbol>();
        foreach (var key in order)
        {
            var members = groups[key];
            double gx0 = members.Min(i => b[i].X0), gy0 = members.Min(i => b[i].Y0);
            double gx1 = members.Max(i => b[i].X1), gy1 = members.Max(i => b[i].Y1);
            if (Math.Max(gx1 - gx0, gy1 - gy0) > 1.25 * maxSize) continue;   // a chain of small strokes (hatching, stairs ...), not one symbol
            foreach (var part in SplitAtDots(members.Select(i => small[i]).ToList()))
            {
                var sym = Describe(part);
                if (sym != null) out_.Add(sym);
            }
        }
        return out_;
    }

    /// <summary>Symbols drawn touching each other (switches stacked on a wall, a legend row of switch types) each have
    /// their own filled dot: two or more separate dots of about the same size -> one symbol per dot, each taking the
    /// lines nearest to it</summary>
    private static List<List<Stroke>> SplitAtDots(List<Stroke> items)
    {
        var dots = items.Where(s => s.Filled).ToList();
        if (dots.Count < 2) return new() { items };
        var b = dots.Select(s => s.BBox).ToArray();
        var tree = new Geo.Tree(b.Select(x => Geo.Env(x.X0, x.Y0, x.X1, x.Y1)));
        var pairs = new List<(int, int)>();
        for (int i = 0; i < b.Length; i++)
            foreach (int j in tree.Query(Geo.Env(b[i].X0, b[i].Y0, b[i].X1, b[i].Y1)))
                pairs.Add((i, j));
        var label = Components(dots.Count, pairs);
        var blobs = new Dictionary<int, List<int>>();
        var order = new List<int>();
        for (int i = 0; i < dots.Count; i++)
        {
            if (!blobs.TryGetValue(label[i], out var l)) { blobs[label[i]] = l = new List<int>(); order.Add(label[i]); }
            l.Add(i);
        }
        if (blobs.Count < 2) return new() { items };
        var boxes = order.Select(k => (X0: blobs[k].Min(i => b[i].X0), Y0: blobs[k].Min(i => b[i].Y0),
                                       X1: blobs[k].Max(i => b[i].X1), Y1: blobs[k].Max(i => b[i].Y1))).ToList();
        var sizes = boxes.Select(x => Math.Max(x.X1 - x.X0, x.Y1 - x.Y0)).ToList();
        if (sizes.Min() < 0.6 * sizes.Max()) return new() { items };   // one dot with a smaller mark, not a row of symbols
        var centres = boxes.Select(x => new Pt((x.X0 + x.X1) / 2, (x.Y0 + x.Y1) / 2)).ToList();
        var parts = boxes.Select(_ => new List<Stroke>()).ToList();
        foreach (var s in items)
        {
            double cx = (s.BBox.X0 + s.BBox.X1) / 2, cy = (s.BBox.Y0 + s.BBox.Y1) / 2;
            int best = 0;
            double bd = double.PositiveInfinity;
            for (int k = 0; k < centres.Count; k++)
            {
                double d = Norm(centres[k].X - cx, centres[k].Y - cy);
                if (d < bd) { bd = d; best = k; }
            }
            parts[best].Add(s);
        }
        return parts.Where(p => p.Count > 0).ToList();
    }

    public static (string? Kind, double Dist) BestMatch(Symbol sym, IList<(Symbol Sym, string Kind)> legend, double sizeWeight = 0.4)
    {
        string? best = null;
        double dist = double.PositiveInfinity;
        foreach (var (reference, kind) in legend)
        {
            double d = ShapeDistance(sym, reference, sizeWeight);
            if (d < dist) { best = kind; dist = d; }
        }
        return dist < MatchLimit ? (best, dist) : (null, dist);
    }

    /// <summary>Unknown symbols with the same shape, most frequent first (used when there is no legend)</summary>
    public static List<List<Symbol>> GroupSameShapes(IList<Symbol> symbols, int minCount = 3)
    {
        var groups = new List<List<Symbol>>();
        foreach (var s in symbols.Take(5000))
        {
            var g = groups.FirstOrDefault(g => ShapeDistance(s, g[0]) < SameShape);
            if (g == null) groups.Add(new List<Symbol> { s });
            else g.Add(s);
        }
        return Py.Sorted(groups.Where(g => g.Count >= minCount), g => g.Count, reverse: true);
    }

    // ── rooms ──
    /// <summary>the outlines (point lists) of a filled path: its pieces joined end to end</summary>
    internal static List<Pt[]> ClosedRuns(IList<Stroke> members)
    {
        var runs = new List<Pt[]>();
        var cur = new List<Pt[]>();
        foreach (var st in members)
        {
            if (st.Pts.Length >= 4 && Math.Max(Math.Abs(st.Pts[0].X - st.Pts[^1].X), Math.Abs(st.Pts[0].Y - st.Pts[^1].Y)) < 0.01)
            {
                runs.Add(st.Pts);                     // a rectangle / closed piece
                continue;
            }
            if (cur.Count > 0)
            {
                var last = cur[^1][^1];
                if (Math.Max(Math.Abs(last.X - st.Pts[0].X), Math.Abs(last.Y - st.Pts[0].Y)) >= 0.01)
                {
                    runs.Add(cur.SelectMany(x => x).ToArray());
                    cur = new List<Pt[]>();
                }
            }
            cur.Add(cur.Count == 0 ? st.Pts : st.Pts[1..]);
        }
        if (cur.Count > 0) runs.Add(cur.SelectMany(x => x).ToArray());
        return runs;
    }
}

/// <summary>
/// Rooms are the enclosed spaces around the room names.
///
/// Wall sets are tried in turn: the walls' own colour (the colour of the big filled shapes - walls drawn solid),
/// then line weights from the heaviest down (walls are drawn heavier than furniture, hatching and dimensions),
/// then everything. Door openings are closed two ways: wall lines are extended up to a door width until they
/// meet another line, and solid walls are thickened by half a door width (and the room grown back by as much).
/// A space is accepted as a room when it has exactly one room name in it, does not reach the edge of the page
/// (the outside of the building), and its area is believable.
/// </summary>
public sealed class RoomFinder
{
    private readonly Polygon _page;
    private readonly (double X0, double Y0, double X1, double Y1) _pageBounds;
    private readonly double _ft;                       // PDF points per foot
    private readonly (double Min, double Max) _areaRange;   // sq ft
    private readonly Pt[] _p0, _p1;
    private readonly double[] _w;
    private readonly bool[] _curve;
    private readonly List<Stroke> _solid;
    private readonly double _eps;
    private readonly List<(bool[] Sel, List<Stroke> Solid)> _sets = new();
    private readonly double[] _gaps;
    private readonly Dictionary<(int, double), Spaces?> _cache = new();
    private readonly Dictionary<int, Geometry> _mass = new();
    private (double X0, double Y0, double X1, double Y1)? _footprint;
    public bool TimedOut;

    private sealed class Spaces
    {
        public List<Geometry> Parts = null!;
        public Geo.Tree Tree = null!;
        public Geometry? Mass;
        public double Grow;
    }

    public RoomFinder(IList<Stroke> items, PRect pageRect, double ft, (double, double) areaRange)
    {
        _page = Geo.Box(pageRect.X0, pageRect.Y0, pageRect.X1, pageRect.Y1);
        _pageBounds = (pageRect.X0, pageRect.Y0, pageRect.X1, pageRect.Y1);
        _ft = ft;
        _areaRange = areaRange;
        // walls are parts of big paths: symbols, letters, hatching and loose furniture bits are left out
        var walls = items.Where(s => !s.Dashed && s.PathSize >= 3 * ft).ToList();
        // solid walls: each filled path is one shape (printer drivers paint them as thin triangles of 2-point
        // pieces); its pieces are long themselves - a filled symbol sharing a big path with others is not a wall
        var byPath = new Dictionary<int, List<Stroke>>();
        var pathOrder = new List<int>();
        foreach (var st in walls)
        {
            if (!st.Filled) continue;
            if (!byPath.TryGetValue(st.PathId, out var l)) { byPath[st.PathId] = l = new List<Stroke>(); pathOrder.Add(st.PathId); }
            l.Add(st);
        }
        var solid = new List<Stroke>();
        foreach (var pid in pathOrder)
        {
            var members = byPath[pid];
            foreach (var run in PdfGeometry.ClosedRuns(members))
            {
                var bb = Stroke.BoxOf(run);
                if (run.Length >= 3 && Math.Max(bb.Item3 - bb.Item1, bb.Item4 - bb.Item2) >= 2 * ft)
                    solid.Add(new Stroke(run, members[0].Width, true, false, false, bb, members[0].Color, members[0].PathSize, members[0].PathId));
            }
        }
        var solidPaths = solid.Select(s => s.PathId).ToHashSet();
        var lines = walls.Where(s => !(s.Filled && solidPaths.Contains(s.PathId))).ToList();
        (_p0, _p1, _w, _curve, _) = PdfGeometry.Segments(lines);
        var segColour = lines.SelectMany(s => Enumerable.Repeat(s.Color, Math.Max(0, s.Pts.Length - 1))).ToArray();
        _solid = solid;
        _eps = Math.Max(0.1 * ft, 0.3);
        // wall sets: (lines to use, solid shapes to use)
        if (solid.Count > 0)
        {
            var colours = new Dictionary<string, int>();
            foreach (var st in solid) colours[st.Color] = (colours.TryGetValue(st.Color, out var c) ? c : 0) + 1;
            var wallColour = Py.MaxBy(colours, kv => kv.Value).Key;
            _sets.Add((segColour.Select(c => c == wallColour).ToArray(), solid.Where(st => st.Color == wallColour).ToList()));
        }
        var widths = _w.Select(v => Py.Round(v, 2)).Distinct().OrderByDescending(v => v).ToList();
        foreach (var t in widths.Count > 1 ? widths.Take(3) : Enumerable.Empty<double>())
            _sets.Add((_w.Select(v => v >= t - 1e-6).ToArray(), solid));
        _sets.Add((Enumerable.Repeat(true, _w.Length).ToArray(), solid));
        _gaps = new[] { 1.5 * ft, 4.0 * ft };            // door openings up to 4 ft are closed
    }

    /// <summary>The box around the building: the biggest connected mass of solid walls (null without solid walls)</summary>
    public (double X0, double Y0, double X1, double Y1)? Footprint()
    {
        if (_sets.Count == 0 || _sets[0].Solid.Count == 0) return null;
        var polys = _sets[0].Solid.Select(st => Geo.Buffer(Geo.Polygon(st.Pts), 0));
        var mass = Geo.BufferUnion(polys, 2 * _ft, quadSegs: 1);
        // a building can fall apart into a few wall masses (open or double-height spaces between them): all big ones
        var parts = Py.Sorted(Geo.Parts(mass), g => -g.Area);
        if (parts.Count == 0) return null;
        var keep = parts.Where(g => g.Area >= 0.15 * parts[0].Area).ToList();
        return Geo.Bounds(Geo.UnionAll(keep));
    }

    private Spaces? GetSpaces(int k, double gap)
    {
        var key = (k, gap);
        if (_cache.TryGetValue(key, out var cached)) return cached;
        var (sel, solid) = _sets[k];
        var idx = Enumerable.Range(0, sel.Length).Where(i => sel[i]).ToArray();
        if (idx.Length + solid.Count == 0 || idx.Length > PdfGeometry.MaxWallSegments || solid.Count > PdfGeometry.MaxWallSegments)
        {
            _cache[key] = null;
            return null;
        }
        var geoms = new List<Geometry>();
        if (idx.Length > 0)
        {
            var p0 = idx.Select(i => _p0[i]).ToArray();
            var p1 = idx.Select(i => _p1[i]).ToArray();
            var curve = idx.Select(i => _curve[i]).ToArray();
            var lines = Enumerable.Range(0, p0.Length).Select(i => Geo.Line(p0[i].X, p0[i].Y, p1[i].X, p1[i].Y)).ToArray();
            var lengths = Enumerable.Range(0, p0.Length).Select(i => { double dx = p1[i].X - p0[i].X, dy = p1[i].Y - p0[i].Y; return Math.Sqrt(dx * dx + dy * dy); }).ToArray();
            // extend straight wall ends across openings
            var ext = Enumerable.Range(0, p0.Length).Where(i => !curve[i] && lengths[i] >= 1.5 * _ft).ToArray();
            var extra = new List<Geometry>();
            if (ext.Length > 0)
            {
                int m = ext.Length;
                var ends = new Pt[2 * m]; var dirs = new Pt[2 * m]; var owner = new int[2 * m];
                for (int q = 0; q < m; q++)
                {
                    int i = ext[q];
                    var u = new Pt((p1[i].X - p0[i].X) / lengths[i], (p1[i].Y - p0[i].Y) / lengths[i]);
                    ends[q] = p1[i]; dirs[q] = u; owner[q] = i;
                    ends[m + q] = p0[i]; dirs[m + q] = new Pt(-u.X, -u.Y); owner[m + q] = i;
                }
                var start = new Pt[2 * m];
                var rays = new LineString[2 * m];
                for (int q = 0; q < 2 * m; q++)
                {
                    start[q] = new Pt(ends[q].X + dirs[q].X * 0.05, ends[q].Y + dirs[q].Y * 0.05);
                    rays[q] = Geo.Line(start[q].X, start[q].Y, ends[q].X + dirs[q].X * gap, ends[q].Y + dirs[q].Y * gap);
                }
                var tree = new Geo.Tree(lines.Select(l => l.EnvelopeInternal));
                var best = Enumerable.Repeat(double.PositiveInfinity, rays.Length).ToArray();
                for (int ri = 0; ri < rays.Length; ri++)
                {
                    foreach (int si in tree.Query(rays[ri].EnvelopeInternal))
                    {
                        if (si == owner[ri] || !rays[ri].Intersects(lines[si])) continue;
                        var c = Geo.Intersection(rays[ri], lines[si]).Centroid;
                        if (c.IsEmpty) continue;
                        var rc = rays[ri].Coordinates;
                        double hit = Geo.LocateOnSegment(rc[0].X, rc[0].Y, rc[1].X, rc[1].Y, c.X, c.Y);
                        if (hit < best[ri]) best[ri] = hit;
                    }
                }
                for (int ri = 0; ri < rays.Length; ri++)
                {
                    if (!double.IsFinite(best[ri])) continue;
                    var tip = new Pt(start[ri].X + dirs[ri].X * (best[ri] + _eps), start[ri].Y + dirs[ri].Y * (best[ri] + _eps));
                    extra.Add(Geo.Line(ends[ri].X, ends[ri].Y, tip.X, tip.Y));
                }
            }
            // each line thickened on its own, then joined: the same shape, far faster than thickening them as one
            // shape when many lines overlap
            geoms.Add(Geo.BufferUnion(lines.Cast<Geometry>().Concat(extra), _eps, quadSegs: 1, cap: EndCapStyle.Square));
        }
        Geometry? mass = null;
        if (solid.Count > 0)
        {
            // solid walls (often painted as many thin slivers): their area, thickened to close door openings
            if (!_mass.TryGetValue(k, out mass))
            {
                var polys = solid.Select(st => Geo.Buffer(Geo.Polygon(st.Pts), 0));
                mass = _mass[k] = Geo.BufferUnion(polys, _eps, quadSegs: 1);
            }
            geoms.Add(Geo.Buffer(mass, gap / 2, quadSegs: 2));
        }
        var walls = Geo.UnionAll(geoms);
        var parts = Geo.Parts(Geo.Difference(_page, walls));
        var result = new Spaces { Parts = parts, Tree = new Geo.Tree(parts.Select(p => p.EnvelopeInternal)), Mass = mass, Grow = solid.Count > 0 ? gap / 2 : 0.0 };
        _cache[key] = result;
        return result;
    }

    private bool Ok(Geometry part, int i, double[] xs, double[] ys, double[] fx, double[] fy)
    {
        var (x0, y0, x1, y1) = Geo.Bounds(part);
        var (px0, py0, px1, py1) = _pageBounds;
        if (x0 <= px0 + 1 || y0 <= py0 + 1 || x1 >= px1 - 1 || y1 >= py1 - 1) return false;   // the outside of the building
        if (_footprint is { } fp && (x0 < fp.X0 - _ft || y0 < fp.Y0 - _ft || x1 > fp.X1 + _ft || y1 > fp.Y1 + _ft))
            return false;                                                                      // reaches past the building's walls
        double sqft = part.Area / Math.Pow(_ft, 2);
        if (!(_areaRange.Min <= sqft && sqft <= _areaRange.Max) || part.Area > 0.5 * _page.Area) return false;
        var loc = Geo.Locator(part);
        for (int k = 0; k < xs.Length; k++)
            if (k != i && Geo.ContainsXY(loc, xs[k], ys[k])) return false;                     // two rooms joined through an open doorway
        int around = 0;
        for (int k = 0; k < xs.Length; k++)
            if (k != i && x0 < xs[k] && xs[k] < x1 && y0 < ys[k] && ys[k] < y1) around++;
        if (around >= 3 && part.Area < 0.5 * (x1 - x0) * (y1 - y0)) return false;            // the open ground wrapped round the house (a porch, a lawn)
        for (int k = 0; k < fx.Length; k++)
            if (Geo.ContainsXY(loc, fx[k], fy[k])) return false;
        return true;
    }

    /// <summary>A room outline per label (x, y, half the text width); null when no enclosed space fits.
    /// The label is tested at its centre and towards both ends, since a name is often written across a line (a door
    /// leaf, furniture); of the spaces it touches the largest acceptable one wins.</summary>
    public List<Geometry?> Find(IList<(double X, double Y, double HW)> labels, IList<(double X, double Y)> forbidden)
    {
        var xs = labels.Select(p => p.X).ToArray();
        var ys = labels.Select(p => p.Y).ToArray();
        var probes = labels.Select(l => new[] { new Pt(l.X, l.Y), new Pt(l.X - 0.6 * l.HW, l.Y), new Pt(l.X + 0.6 * l.HW, l.Y) }).ToList();
        var fx = forbidden.Select(p => p.X).ToArray();
        var fy = forbidden.Select(p => p.Y).ToArray();
        var out_ = new List<Geometry?>(new Geometry?[labels.Count]);
        var todo = new SortedSet<int>(Enumerable.Range(0, labels.Count));
        _footprint = Footprint();
        var clock = Stopwatch.StartNew();
        TimedOut = false;
        for (int k = 0; k < _sets.Count; k++)
        {
            foreach (var gap in _gaps)
            {
                if (todo.Count > 0 && clock.Elapsed.TotalSeconds > PdfGeometry.RoomTimeLimit)
                {
                    TimedOut = true;
                    return out_;
                }
                if (todo.Count == 0) return out_;
                var spaces = GetSpaces(k, gap);
                if (spaces == null) continue;
                foreach (int i in todo.ToList())
                {
                    var hits = new SortedSet<int>();
                    foreach (var p in probes[i])
                        foreach (int j in spaces.Tree.Query(Geo.Env(p.X, p.Y, p.X, p.Y)))
                            if (Geo.ContainsXY(Geo.Locator(spaces.Parts[j]), p.X, p.Y)) hits.Add(j);
                    if (hits.Count == 0)        // the whole label sits on lines: the spaces right next to it
                        foreach (var p in probes[i])
                        {
                            var pt = Geo.Point(p.X, p.Y);
                            foreach (int j in spaces.Tree.Query(Geo.Env(p.X - _ft, p.Y - _ft, p.X + _ft, p.Y + _ft)))
                                if (spaces.Parts[j].IsWithinDistance(pt, _ft)) hits.Add(j);
                        }
                    foreach (int j in Py.Sorted(hits, j => -spaces.Parts[j].Area))
                    {
                        if (!Ok(spaces.Parts[j], i, xs, ys, fx, fy)) continue;
                        var shell = ((Polygon)spaces.Parts[j]).Shell;
                        Geometry room = Geo.Buffer(Geo.F.CreatePolygon(shell.Coordinates), _eps + spaces.Grow, join: JoinStyle.Mitre);
                        if (spaces.Mass != null) room = Geo.Difference(room, spaces.Mass);      // grown back up to the wall faces
                        if (room is not Polygon)
                        {
                            // (nothing left once the walls are taken off: not a room - the next space is tried)
                            var pieces = Geo.Parts(room).OfType<Polygon>().Where(g => g.Area > 0).ToList();
                            if (pieces.Count == 0) continue;
                            room = Py.MaxBy(pieces, g => g.Area);
                        }
                        out_[i] = room;
                        todo.Remove(i);
                        break;
                    }
                }
            }
        }
        return out_;
    }
}
