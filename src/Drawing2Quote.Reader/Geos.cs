using System.Runtime.InteropServices;
using NetTopologySuite.Geometries;
using NetTopologySuite.IO;
using NetTopologySuite.Operation.Buffer;

namespace Drawing2Quote.Reader;

/// <summary>
/// The geometry operations whose results depend on the algorithm - buffer, union, difference, intersection,
/// simplification, point on surface - worked out by GEOS 3.14 (the C++ library shapely runs on, native/geos_c.dll),
/// so traced room outlines and their areas come out as the Python reader's. Geometries cross over as WKB (exact
/// doubles); NetTopologySuite holds them and answers the predicates.
/// </summary>
internal static unsafe class Geos
{
    private const string Lib = "geos_c";

    [DllImport(Lib)] private static extern IntPtr GEOS_init_r();
    [DllImport(Lib)] private static extern IntPtr GEOSWKBReader_create_r(IntPtr h);
    [DllImport(Lib)] private static extern IntPtr GEOSWKBReader_read_r(IntPtr h, IntPtr reader, byte* wkb, nuint size);
    [DllImport(Lib)] private static extern IntPtr GEOSWKBWriter_create_r(IntPtr h);
    [DllImport(Lib)] private static extern byte* GEOSWKBWriter_write_r(IntPtr h, IntPtr writer, IntPtr g, out nuint size);
    [DllImport(Lib)] private static extern void GEOSFree_r(IntPtr h, IntPtr buffer);
    [DllImport(Lib)] private static extern void GEOSGeom_destroy_r(IntPtr h, IntPtr g);
    [DllImport(Lib)] private static extern IntPtr GEOSBufferWithStyle_r(IntPtr h, IntPtr g, double width, int quadsegs, int endCapStyle, int joinStyle, double mitreLimit);
    [DllImport(Lib)] private static extern IntPtr GEOSUnaryUnion_r(IntPtr h, IntPtr g);
    [DllImport(Lib)] private static extern IntPtr GEOSDifference_r(IntPtr h, IntPtr a, IntPtr b);
    [DllImport(Lib)] private static extern IntPtr GEOSIntersection_r(IntPtr h, IntPtr a, IntPtr b);
    [DllImport(Lib)] private static extern IntPtr GEOSTopologyPreserveSimplify_r(IntPtr h, IntPtr g, double tolerance);
    [DllImport(Lib)] private static extern IntPtr GEOSPointOnSurface_r(IntPtr h, IntPtr g);
    [DllImport(Lib)] private static extern IntPtr GEOSGetCentroid_r(IntPtr h, IntPtr g);
    [DllImport(Lib)] private static extern int GEOSGeomGetX_r(IntPtr h, IntPtr g, out double x);
    [DllImport(Lib)] private static extern int GEOSGeomGetY_r(IntPtr h, IntPtr g, out double y);
    [DllImport(Lib)] private static extern IntPtr GEOSGeom_createCollection_r(IntPtr h, int type, IntPtr* geoms, uint ngeoms);
    [DllImport(Lib)] private static extern IntPtr GEOSGeom_createEmptyCollection_r(IntPtr h, int type);
    [DllImport(Lib)] private static extern sbyte GEOSisValid_r(IntPtr h, IntPtr g);

    private const int GeometryCollection = 7;

    /// <summary>a GEOS context per thread (the reentrant API)</summary>
    private sealed class Ctx
    {
        public readonly IntPtr H, Reader, Writer;

        public Ctx()
        {
            H = GEOS_init_r();
            Reader = GEOSWKBReader_create_r(H);
            Writer = GEOSWKBWriter_create_r(H);
        }
    }

    [ThreadStatic] private static Ctx? _ctx;
    private static Ctx C => _ctx ??= new Ctx();

    [ThreadStatic] private static WKBWriter? _ntsWriter;
    [ThreadStatic] private static WKBReader? _ntsReader;
    private static WKBWriter NtsWriter => _ntsWriter ??= new WKBWriter();
    private static WKBReader NtsReader => _ntsReader ??= new WKBReader(Geo.Services);

    private static IntPtr ToGeos(Geometry g)
    {
        var wkb = NtsWriter.Write(g);
        fixed (byte* p = wkb)
        {
            var r = GEOSWKBReader_read_r(C.H, C.Reader, p, (nuint)wkb.Length);
            if (r == IntPtr.Zero) throw new InvalidOperationException("GEOS could not read a geometry");
            return r;
        }
    }

    /// <summary>the GEOS geometry as a NetTopologySuite one (the GEOS one is destroyed)</summary>
    private static Geometry FromGeos(IntPtr g)
    {
        if (g == IntPtr.Zero) throw new InvalidOperationException("GEOS operation failed");
        try
        {
            var buf = GEOSWKBWriter_write_r(C.H, C.Writer, g, out var size);
            try
            {
                var bytes = new byte[(int)size];
                Marshal.Copy((IntPtr)buf, bytes, 0, bytes.Length);
                return NtsReader.Read(bytes);
            }
            finally
            {
                GEOSFree_r(C.H, (IntPtr)buf);
            }
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, g);
        }
    }

    private static int Cap(EndCapStyle c) => c switch { EndCapStyle.Flat => 2, EndCapStyle.Square => 3, _ => 1 };
    private static int Join(JoinStyle j) => j switch { JoinStyle.Mitre => 2, JoinStyle.Bevel => 3, _ => 1 };

    public static Geometry Buffer(Geometry g, double d, int quadSegs, EndCapStyle cap, JoinStyle join, double mitreLimit)
    {
        var a = ToGeos(g);
        try
        {
            return FromGeos(GEOSBufferWithStyle_r(C.H, a, d, quadSegs, Cap(cap), Join(join), mitreLimit));
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, a);
        }
    }

    /// <summary>union_all of the given GEOS geometries (owned: the collection takes them)</summary>
    private static Geometry UnionOwned(List<IntPtr> parts)
    {
        IntPtr coll;
        if (parts.Count == 0) coll = GEOSGeom_createEmptyCollection_r(C.H, GeometryCollection);
        else
        {
            var arr = parts.ToArray();
            fixed (IntPtr* p = arr) coll = GEOSGeom_createCollection_r(C.H, GeometryCollection, p, (uint)arr.Length);
        }
        try
        {
            return FromGeos(GEOSUnaryUnion_r(C.H, coll));
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, coll);
        }
    }

    public static Geometry UnionAll(IEnumerable<Geometry> geoms) =>
        UnionOwned(geoms.Where(g => g != null).Select(ToGeos).ToList());

    public static Geometry BufferUnion(IEnumerable<Geometry> geoms, double d, int quadSegs, EndCapStyle cap, JoinStyle join, double mitreLimit)
    {
        var parts = new List<IntPtr>();
        foreach (var g in geoms)
        {
            var a = ToGeos(g);
            var b = GEOSBufferWithStyle_r(C.H, a, d, quadSegs, Cap(cap), Join(join), mitreLimit);
            GEOSGeom_destroy_r(C.H, a);
            if (b == IntPtr.Zero) throw new InvalidOperationException("GEOS buffer failed");
            parts.Add(b);
        }
        return UnionOwned(parts);
    }

    public static Geometry Binary(Geometry a, Geometry b, bool intersection)
    {
        var ga = ToGeos(a);
        var gb = ToGeos(b);
        try
        {
            return FromGeos(intersection ? GEOSIntersection_r(C.H, ga, gb) : GEOSDifference_r(C.H, ga, gb));
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, ga);
            GEOSGeom_destroy_r(C.H, gb);
        }
    }

    public static Geometry Simplify(Geometry g, double tolerance)
    {
        var a = ToGeos(g);
        try
        {
            return FromGeos(GEOSTopologyPreserveSimplify_r(C.H, a, tolerance));
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, a);
        }
    }

    public static bool IsValid(Geometry g)
    {
        var a = ToGeos(g);
        try
        {
            return GEOSisValid_r(C.H, a) == 1;
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, a);
        }
    }

    public static Coordinate PointOnSurface(Geometry g, bool centroid)
    {
        var a = ToGeos(g);
        var p = centroid ? GEOSGetCentroid_r(C.H, a) : GEOSPointOnSurface_r(C.H, a);
        try
        {
            if (p == IntPtr.Zero) throw new InvalidOperationException("GEOS operation failed");
            GEOSGeomGetX_r(C.H, p, out var x);
            GEOSGeomGetY_r(C.H, p, out var y);
            return new Coordinate(x, y);
        }
        finally
        {
            GEOSGeom_destroy_r(C.H, p);
            GEOSGeom_destroy_r(C.H, a);
        }
    }
}
