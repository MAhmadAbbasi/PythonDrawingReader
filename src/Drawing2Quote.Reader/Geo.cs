using NetTopologySuite;
using NetTopologySuite.Algorithm.Locate;
using NetTopologySuite.Geometries;
using NetTopologySuite.Index.Strtree;
using NetTopologySuite.Operation.Buffer;
using NetTopologySuite.Operation.OverlayNG;

namespace Drawing2Quote.Reader;

public readonly record struct Pt(double X, double Y);

/// <summary>shapely's functions on NetTopologySuite (the Java original of GEOS, which shapely runs on), with the
/// OverlayNG engine GEOS uses</summary>
internal static class Geo
{
    public static readonly NtsGeometryServices Services = new(GeometryOverlay.NG);
    public static readonly GeometryFactory F = Services.CreateGeometryFactory();

    /// <summary>shapely.box(x0, y0, x1, y1) (counter-clockwise from the lower right corner)</summary>
    public static Polygon Box(double x0, double y0, double x1, double y1) =>
        F.CreatePolygon(new[] { new Coordinate(x1, y0), new Coordinate(x1, y1), new Coordinate(x0, y1), new Coordinate(x0, y0), new Coordinate(x1, y0) });

    public static Polygon Box(PRect r) => Box(r.X0, r.Y0, r.X1, r.Y1);

    public static LineString Line(double x0, double y0, double x1, double y1) =>
        F.CreateLineString(new[] { new Coordinate(x0, y0), new Coordinate(x1, y1) });

    public static LineString Line(IReadOnlyList<Pt> pts) => F.CreateLineString(pts.Select(p => new Coordinate(p.X, p.Y)).ToArray());

    /// <summary>shapely.Polygon(points): the ring closed if it isn't</summary>
    public static Polygon Polygon(IReadOnlyList<Pt> pts)
    {
        var c = pts.Select(p => new Coordinate(p.X, p.Y)).ToList();
        if (c.Count > 0 && !c[0].Equals2D(c[^1])) c.Add(c[0].Copy());
        while (c.Count < 4 && c.Count > 0) c.Add(c[0].Copy());
        return F.CreatePolygon(c.ToArray());
    }

    public static Polygon Polygon(IReadOnlyList<Pt> shell, IEnumerable<IReadOnlyList<Pt>> holes)
    {
        LinearRing Ring(IReadOnlyList<Pt> pts)
        {
            var c = pts.Select(p => new Coordinate(p.X, p.Y)).ToList();
            if (c.Count > 0 && !c[0].Equals2D(c[^1])) c.Add(c[0].Copy());
            while (c.Count < 4 && c.Count > 0) c.Add(c[0].Copy());
            return F.CreateLinearRing(c.ToArray());
        }
        return F.CreatePolygon(Ring(shell), holes.Select(Ring).ToArray());
    }

    public static NetTopologySuite.Geometries.Point Point(double x, double y) => F.CreatePoint(new Coordinate(x, y));

    /// <summary>shapely.buffer(g, d, quad_segs, cap_style, join_style, mitre_limit) - worked out by GEOS, as shapely does</summary>
    public static Geometry Buffer(Geometry g, double d, int quadSegs = 8, EndCapStyle cap = EndCapStyle.Round,
                                  JoinStyle join = JoinStyle.Round, double mitreLimit = 5.0) =>
        Geos.Buffer(g, d, quadSegs, cap, join, mitreLimit);

    /// <summary>shapely.union_all(shapely.buffer(geoms, ...)): each one thickened, then all joined (in GEOS)</summary>
    public static Geometry BufferUnion(IEnumerable<Geometry> geoms, double d, int quadSegs = 8, EndCapStyle cap = EndCapStyle.Round,
                                       JoinStyle join = JoinStyle.Round, double mitreLimit = 5.0) =>
        Geos.BufferUnion(geoms, d, quadSegs, cap, join, mitreLimit);

    /// <summary>shapely.union_all</summary>
    public static Geometry UnionAll(IEnumerable<Geometry> geoms) => Geos.UnionAll(geoms);

    /// <summary>a - b, a & b (GEOS: OverlayNG with its robust fallbacks)</summary>
    public static Geometry Difference(Geometry a, Geometry b) => Geos.Binary(a, b, intersection: false);

    public static Geometry Intersection(Geometry a, Geometry b) => Geos.Binary(a, b, intersection: true);

    /// <summary>g.simplify(tolerance) (topology preserving)</summary>
    public static Geometry Simplify(Geometry g, double tolerance) => Geos.Simplify(g, tolerance);

    /// <summary>g.is_valid</summary>
    public static bool IsValid(Geometry g) => Geos.IsValid(g);

    /// <summary>g.representative_point()</summary>
    public static Coordinate InteriorPoint(Geometry g) => Geos.PointOnSurface(g, centroid: false);

    /// <summary>g.centroid</summary>
    public static Coordinate Centroid(Geometry g) => Geos.PointOnSurface(g, centroid: true);

    /// <summary>shapely.get_parts: the parts of a multi geometry or collection (the geometry itself otherwise)</summary>
    public static List<Geometry> Parts(Geometry g)
    {
        var out_ = new List<Geometry>();
        if (g.IsEmpty) return out_;
        if (g is GeometryCollection gc)
            for (int i = 0; i < gc.NumGeometries; i++) out_.Add(gc.GetGeometryN(i));
        else out_.Add(g);
        return out_;
    }

    public static (double X0, double Y0, double X1, double Y1) Bounds(Geometry g)
    {
        var e = g.EnvelopeInternal;
        return (e.MinX, e.MinY, e.MaxX, e.MaxY);
    }

    /// <summary>shapely.contains_xy: the point is in the interior (not on the boundary)</summary>
    public static bool ContainsXY(IPointOnGeometryLocator loc, double x, double y) => loc.Locate(new Coordinate(x, y)) == Location.Interior;

    public static IPointOnGeometryLocator Locator(Geometry g) => new IndexedPointInAreaLocator(g);

    /// <summary>an STRtree over envelopes; queries give the candidate indexes in ascending order</summary>
    public sealed class Tree
    {
        private readonly STRtree<int> _t = new();
        public readonly int Count;

        public Tree(IEnumerable<Envelope> envelopes)
        {
            int i = 0;
            foreach (var e in envelopes) _t.Insert(e, i++);
            Count = i;
            if (i > 0) _t.Build();
        }

        public List<int> Query(Envelope e)
        {
            if (Count == 0) return new List<int>();
            var r = _t.Query(e).ToList();
            r.Sort();
            return r;
        }
    }

    public static Envelope Env(double x0, double y0, double x1, double y1) => new(x0, x1, y0, y1);

    /// <summary>shapely.line_locate_point for a two-point line: how far along it the point's projection lies</summary>
    public static double LocateOnSegment(double x0, double y0, double x1, double y1, double px, double py)
    {
        double dx = x1 - x0, dy = y1 - y0;
        double len2 = dx * dx + dy * dy;
        double length = Math.Sqrt(len2);
        double r;
        if (px == x0 && py == y0) r = 0.0;
        else if (px == x1 && py == y1) r = 1.0;
        else if (len2 <= 0) return 0.0;
        else r = ((px - x0) * dx + (py - y0) * dy) / len2;
        if (r <= 0) return 0.0;
        if (r <= 1) return r * length;
        return length;
    }
}
