using MuPDF.NET;

namespace Drawing2Quote.Reader;

/// <summary>A copy of the PDF with what was read drawn on top, to check the reading by eye: room outlines (coloured,
/// with the name), every counted symbol (a box with its type), the legend area, and - on pages without a legend -
/// the repeated symbol shapes that were found but not counted (grey, lettered).</summary>
public static class Overlay
{
    private static readonly Dictionary<string, string> Short = new()
    {
        ["light"] = "L", ["fan"] = "F", ["exhaust_fan"] = "EF", ["socket_5A"] = "5A", ["socket_15A"] = "15A", ["switch"] = "SW",
        ["ac_point"] = "AC", ["tv_point"] = "TV", ["data_point"] = "D", ["curtain"] = "C", ["geyser"] = "G", ["db_panel"] = "DB",
    };

    private static readonly uint[] CrcTable = Enumerable.Range(0, 256).Select(n =>
    {
        uint c = (uint)n;
        for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        return c;
    }).ToArray();

    private static uint Crc32(byte[] data)
    {
        uint c = 0xFFFFFFFFu;
        foreach (var b in data) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    /// <summary>colorsys.hls_to_rgb</summary>
    private static float[] Hls(double h, double l, double s)
    {
        double V(double m1, double m2, double hue)
        {
            hue = Py.Mod(hue, 1.0);
            if (hue < 1.0 / 6) return m1 + (m2 - m1) * hue * 6.0;
            if (hue < 0.5) return m2;
            if (hue < 2.0 / 3) return m1 + (m2 - m1) * (2.0 / 3 - hue) * 6.0;
            return m1;
        }
        if (s == 0.0) return new[] { (float)l, (float)l, (float)l };
        double m2 = l <= 0.5 ? l * (1.0 + s) : l + s - l * s;
        double m1 = 2.0 * l - m2;
        return new[] { (float)V(m1, m2, h + 1.0 / 3), (float)V(m1, m2, h), (float)V(m1, m2, h - 1.0 / 3) };
    }

    private static float[] Colour(string name, double light = 0.45) =>
        Hls((Crc32(System.Text.Encoding.UTF8.GetBytes(name)) % 360) / 360.0, light, 0.85);

    private static Point P(double x, double y, Mat2D m)
    {
        var (a, b) = m.Apply(x, y);
        return new Point((float)a, (float)b);
    }

    private static Rect R(PRect r, Mat2D m)
    {
        var t = r.Transform(m);
        return new Rect((float)t.X0, (float)t.Y0, (float)t.X1, (float)t.Y1);
    }

    /// <summary>the drawing (a PDF, or the traced drawing of a picture) with the readings of each sheet drawn on it</summary>
    public static byte[] Draw(byte[] data, IList<Sheet> sheets)
    {
        using var doc = new PdfDoc(data);
        lock (PdfDoc.Lock)
        {
            foreach (var sheet in sheets)
            {
                if (sheet.Debug.Page is not { } pno) continue;
                var page = doc.Raw[pno];
                var mm = page.DerotationMatrix;             // readings are in the page as shown; drawing is in the unrotated page
                var m = new Mat2D(mm.A, mm.B, mm.C, mm.D, mm.E, mm.F);
                var s = page.NewShape();
                for (int i = 0; i < sheet.Rooms.Count; i++)
                {
                    var room = sheet.Rooms[i];
                    var col = Colour(room.Name + i, 0.5);
                    if (room.Poly is NetTopologySuite.Geometries.Polygon poly)
                    {
                        var pts = poly.ExteriorRing.Coordinates.Select(c => P(c.X, c.Y, m)).ToArray();
                        s.DrawPolyline(pts);
                        s.Finish(color: col, fill: col, fillOpacity: 0.18f, width: 1.2f, closePath: true);
                    }
                    else if (room.Poly == null)
                    {
                        s.DrawCircle(P(room.X, room.Y, m), 6);
                        s.Finish(color: col, width: 1.2f, dashes: "[2 2] 0");
                    }
                }
                foreach (var p in sheet.Points)
                {
                    s.DrawRect(R(new PRect(p.X - 5, p.Y - 5, p.X + 5, p.Y + 5), m));
                    s.Finish(color: Colour(p.Kind, 0.4), width: 0.8f);
                }
                if (sheet.Debug.Legend is { } legend)
                {
                    s.DrawRect(R(legend, m));
                    s.Finish(color: new[] { 0f, 0.4f, 1f }, width: 1, dashes: "[4 2] 0");
                }
                foreach (var group in sheet.Debug.Unknown)
                    foreach (var sym in group)
                    {
                        s.DrawRect(R(PRect.From(sym.BBox), m));
                        s.Finish(color: new[] { 0.5f, 0.5f, 0.5f }, width: 0.6f, dashes: "[1 1] 0");
                    }
                s.Commit();
                for (int i = 0; i < sheet.Rooms.Count; i++)
                {
                    var room = sheet.Rooms[i];
                    var label = room.Name + (room.AreaSqft is { } a && a != 0 ? $"  {Py.G(a)} sq ft" : "");
                    page.InsertText(P(room.X - 2 * label.Length, room.Y - 8, m), label, fontSize: 6, color: Colour(room.Name + i, 0.3), rotate: page.Rotation);
                }
                foreach (var p in sheet.Points)
                    page.InsertText(P(p.X + 5.5, p.Y - 5, m), Short.TryGetValue(p.Kind, out var sh) ? sh : p.Kind, fontSize: 4.5f,
                                    color: Colour(p.Kind, 0.35), rotate: page.Rotation);
                for (int g = 0; g < sheet.Debug.Unknown.Count; g++)
                    foreach (var sym in sheet.Debug.Unknown[g])
                        page.InsertText(P(sym.BBox.X1 + 1, sym.BBox.Y0, m), ((char)(65 + g % 26)).ToString(), fontSize: 4.5f,
                                        color: new[] { 0.4f, 0.4f, 0.4f }, rotate: page.Rotation);
            }
        }
        return doc.ToBytes(3, true);
    }
}
