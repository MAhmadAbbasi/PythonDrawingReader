using MuPDF.NET;
using OpenCvSharp;

namespace Drawing2Quote.Reader;

/// <summary>A PDF opened with MuPDF (the engine PyMuPDF runs on), giving what the readers use in PyMuPDF's form</summary>
internal sealed class PdfDoc : IDisposable
{
    /// <summary>PyMuPDF 2.0 gives text boxes as MuPDF measures them (no ascender / descender corrections)</summary>
    public const int TextFlagsDict = 199;      // pymupdf.TEXTFLAGS_DICT

    private static readonly object InitLock = new();
    private static bool _init;

    internal static readonly object Lock = new();   // MuPDF calls one at a time (the readers' own work runs in parallel)

    private readonly Document _doc;

    private static void Init()
    {
        lock (InitLock)
        {
            if (_init) return;
            Tools.UnsetQuadCorrections(true);
            _init = true;
        }
    }

    public PdfDoc(byte[] data)
    {
        Init();
        lock (Lock) _doc = new Document(data, "pdf");
    }

    /// <summary>a new, empty PDF</summary>
    public PdfDoc()
    {
        Init();
        lock (Lock) _doc = new Document();
    }

    public int PageCount { get { lock (Lock) return _doc.PageCount; } }
    public bool NeedsPass { get { lock (Lock) return _doc.NeedsPass; } }
    public bool IsEncrypted { get { lock (Lock) return _doc.IsEncrypted; } }
    internal Document Raw => _doc;

    public PdfPage this[int i] { get { lock (Lock) return new PdfPage(this, _doc[i], i); } }

    public int PageImageCount(int pno) { lock (Lock) return _doc.GetPageImages(pno).Count; }

    public PdfPage NewPage(double width, double height)
    {
        lock (Lock)
        {
            var p = _doc.NewPage(-1, (float)width, (float)height);
            return new PdfPage(this, p, _doc.PageCount - 1);
        }
    }

    /// <summary>the page's drawing commands replaced by this content stream</summary>
    public void SetContents(PdfPage page, byte[] content)
    {
        lock (Lock)
        {
            int xref = _doc.GetNewXref();
            _doc.UpdateObject(xref, "<<>>");
            _doc.UpdateStream(xref, content, true);
            _doc.XrefSetKey(page.Raw.Xref, "Contents", $"{xref} 0 R");
        }
    }

    public byte[] ToBytes(int garbage, bool deflate)
    {
        lock (Lock)
        {
            using var ms = new MemoryStream();
            _doc.Save(ms, garbage: garbage, deflate: deflate ? 1 : 0);
            return ms.ToArray();
        }
    }

    public void Dispose()
    {
        lock (Lock) _doc.Close();
    }
}

internal sealed class PdfPage
{
    private readonly PdfDoc _owner;
    internal readonly Page Raw;
    public readonly int Number;
    public readonly PRect Rect;
    public readonly int Rotation;
    public readonly Mat2D RotationMatrix, DerotationMatrix;

    internal PdfPage(PdfDoc owner, Page page, int number)
    {
        _owner = owner;
        Raw = page;
        Number = number;
        var r = page.Rect;
        Rect = new PRect(r.X0, r.Y0, r.X1, r.Y1);
        Rotation = page.Rotation;
        RotationMatrix = M(page.RotationMatrix);
        DerotationMatrix = M(page.DerotationMatrix);
    }

    private static Mat2D M(Matrix m) => new(m.A, m.B, m.C, m.D, m.E, m.F);

    private static Pt P(MuPDF.NET.Point p) => new(p.X, p.Y);

    /// <summary>page.get_drawings()</summary>
    public List<Drawing> GetDrawings()
    {
        List<PathInfo> paths;
        lock (PdfDoc.Lock) paths = Raw.GetDrawings();
        var out_ = new List<Drawing>(paths.Count);
        foreach (var d in paths)
        {
            var dr = new Drawing
            {
                Type = d.Type ?? "",
                Width = d.Width,
                Fill = d.Fill is { Length: > 0 } f ? f.Select(v => (double)v).ToArray() : null,
                Color = d.Color is { Length: > 0 } c ? c.Select(v => (double)v).ToArray() : null,
                Dashes = d.Dashes,
                FillOpacity = d.FillOpacity,
                Rect = new PRect(d.Rect.X0, d.Rect.Y0, d.Rect.X1, d.Rect.Y1),
            };
            foreach (var it in d.Items)
            {
                switch (it.Type)
                {
                    case "l": dr.Items.Add(new DrawItem("l", P(it.P1), P(it.LastPoint))); break;
                    case "c": dr.Items.Add(new DrawItem("c", P(it.P1), P(it.P2), P(it.P3), P(it.LastPoint))); break;
                    case "re": dr.Items.Add(new DrawItem("re", new Pt(it.Rect.X0, it.Rect.Y0), new Pt(it.Rect.X1, it.Rect.Y1))); break;
                    case "qu": dr.Items.Add(new DrawItem("qu", P(it.Quad.UL), P(it.Quad.UR), P(it.Quad.LL), P(it.Quad.LR))); break;
                }
            }
            out_.Add(dr);
        }
        return out_;
    }

    /// <summary>the text spans of page.get_text("dict"), line by line (unrotated page coordinates)</summary>
    public List<List<(string Text, PRect BBox, double Size)>> TextLines()
    {
        var out_ = new List<List<(string, PRect, double)>>();
        PageInfo? info;
        lock (PdfDoc.Lock) info = Raw.GetText("dict", flags: PdfDoc.TextFlagsDict) as PageInfo;
        if (info?.Blocks == null) return out_;
        foreach (var b in info.Blocks)
        {
            if (b.Lines == null) continue;
            foreach (var l in b.Lines)
            {
                var spans = new List<(string, PRect, double)>();
                foreach (var s in l.Spans ?? new List<Span>())
                    spans.Add((s.Text ?? "", new PRect(s.Bbox.X0, s.Bbox.Y0, s.Bbox.X1, s.Bbox.Y1), s.Size));
                out_.Add(spans);
            }
        }
        return out_;
    }

    /// <summary>page.get_text("text")</summary>
    public string PlainText()
    {
        lock (PdfDoc.Lock) return Raw.GetText("text") as string ?? "";
    }

    /// <summary>page.get_image_info(): (pixel width, box on the page) of each picture</summary>
    public List<(int Width, PRect BBox)> ImageInfo()
    {
        lock (PdfDoc.Lock)
        {
            var info = Raw.GetImageInfo(false, false);
            var out_ = new List<(int, PRect)>();
            foreach (var b in info)
                out_.Add((b.Width, new PRect(b.Bbox.X0, b.Bbox.Y0, b.Bbox.X1, b.Bbox.Y1)));
            return out_;
        }
    }

    /// <summary>page.get_pixmap(dpi=dpi, alpha=False) as an RGB image</summary>
    public Mat RenderRgb(double dpi)
    {
        lock (PdfDoc.Lock)
        {
            float zoom = (float)(dpi / 72);
            using var pix = Raw.GetPixmap(new Matrix(zoom, zoom), alpha: false);
            int w = pix.Width, h = pix.Height, n = pix.N, stride = pix.Stride;
            var samples = pix.Samples;
            var mat = new Mat(h, w, MatType.CV_8UC3);
            var idx = mat.GetGenericIndexer<Vec3b>();
            for (int y = 0; y < h; y++)
            {
                int row = y * stride;
                for (int x = 0; x < w; x++)
                {
                    int o = row + x * n;
                    idx[y, x] = n >= 3 ? new Vec3b(samples[o], samples[o + 1], samples[o + 2]) : new Vec3b(samples[o], samples[o], samples[o]);
                }
            }
            return mat;
        }
    }
}
