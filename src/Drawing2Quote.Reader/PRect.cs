namespace Drawing2Quote.Reader;

/// <summary>
/// A rectangle with PyMuPDF's Rect behaviour, which the PDF reader's rules were written against: union ignores an
/// empty rectangle, a point is inside when x0 &lt;= x &lt; x1 (MuPDF), unions and intersections are worked out in
/// MuPDF's single precision.
/// </summary>
public readonly record struct PRect(double X0, double Y0, double X1, double Y1)
{
    private const double MinInf = -2147483648.0, MaxInf = 2147483520.0;

    public double Width => Math.Max(0, X1 - X0);
    public double Height => Math.Max(0, Y1 - Y0);
    public bool IsEmpty => X0 >= X1 || Y0 >= Y1;
    public bool IsInfinite => X0 == MinInf && Y0 == MinInf && X1 == MaxInf && Y1 == MaxInf;
    public double Area => Width * Height;
    public double CX => (X0 + X1) / 2;
    public double CY => (Y0 + Y1) / 2;

    public static PRect From((double, double, double, double) b) => new(b.Item1, b.Item2, b.Item3, b.Item4);
    public (double, double, double, double) Tuple => (X0, Y0, X1, Y1);

    private static double F(double v) => (float)v;

    /// <summary>r | other (Rect.include_rect)</summary>
    public PRect Union(PRect r)
    {
        if (r.IsInfinite || IsInfinite) return new PRect(MinInf, MinInf, MaxInf, MaxInf);
        if (r.IsEmpty) return this;
        if (IsEmpty) return r;
        return new PRect(F(Math.Min(F(X0), F(r.X0))), F(Math.Min(F(Y0), F(r.Y0))),
                         F(Math.Max(F(X1), F(r.X1))), F(Math.Max(F(Y1), F(r.Y1))));
    }

    public static PRect operator |(PRect a, PRect b) => a.Union(b);

    /// <summary>r &amp; other (Rect.intersect)</summary>
    public PRect Intersect(PRect r)
    {
        if (r.IsInfinite) return this;
        if (IsInfinite) return r;
        if (r.IsEmpty) return r;
        if (IsEmpty) return this;
        return new PRect(F(Math.Max(F(X0), F(r.X0))), F(Math.Max(F(Y0), F(r.Y0))),
                         F(Math.Min(F(X1), F(r.X1))), F(Math.Min(F(Y1), F(r.Y1))));
    }

    public static PRect operator &(PRect a, PRect b) => a.Intersect(b);

    public bool Intersects(PRect r) =>
        !IsEmpty && !IsInfinite && !r.IsEmpty && !r.IsInfinite && X0 < r.X1 && r.X0 < X1 && Y0 < r.Y1 && r.Y0 < Y1;

    /// <summary>a point inside (MuPDF: the right and bottom edges are outside)</summary>
    public bool Contains(double x, double y)
    {
        float px = (float)x, py = (float)y;
        return px >= (float)X0 && px < (float)X1 && py >= (float)Y0 && py < (float)Y1;
    }

    /// <summary>a rectangle inside (edges included)</summary>
    public bool Contains(PRect r) => X0 <= r.X0 && r.X0 <= r.X1 && r.X1 <= X1 && Y0 <= r.Y0 && r.Y0 <= r.Y1 && r.Y1 <= Y1;

    /// <summary>r + (a, b, c, d)</summary>
    public PRect Add(double a, double b, double c, double d) => new(X0 + a, Y0 + b, X1 + c, Y1 + d);

    public PRect Transform(Mat2D m)
    {
        if (IsInfinite) return this;
        if (X0 > X1 || Y0 > Y1) return this;        // (fz_transform_rect leaves an invalid rectangle)
        var (ax, ay) = m.Apply(X0, Y0);
        var (bx, by) = m.Apply(X1, Y0);
        var (cx, cy) = m.Apply(X0, Y1);
        var (dx, dy) = m.Apply(X1, Y1);
        return new PRect(F(Math.Min(Math.Min(ax, bx), Math.Min(cx, dx))), F(Math.Min(Math.Min(ay, by), Math.Min(cy, dy))),
                         F(Math.Max(Math.Max(ax, bx), Math.Max(cx, dx))), F(Math.Max(Math.Max(ay, by), Math.Max(cy, dy))));
    }
}

/// <summary>A PDF matrix (a b c d e f): x' = a x + c y + e, y' = b x + d y + f</summary>
public readonly record struct Mat2D(double A, double B, double C, double D, double E, double F)
{
    public static readonly Mat2D Identity = new(1, 0, 0, 1, 0, 0);

    /// <summary>in MuPDF's single precision (pymupdf.Point * Matrix)</summary>
    public (double, double) Apply(double x, double y)
    {
        float fx = (float)x, fy = (float)y;
        return ((float)(fx * (float)A + fy * (float)C + (float)E), (float)(fx * (float)B + fy * (float)D + (float)F));
    }
}
