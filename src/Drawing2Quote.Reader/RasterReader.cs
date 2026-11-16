using System.Text;
using System.Text.RegularExpressions;
using NetTopologySuite.Geometries;
using OpenCvSharp;
using CvPoint = OpenCvSharp.Point;
using CvRect = OpenCvSharp.Rect;

namespace Drawing2Quote.Reader;

/// <summary>
/// Scanned PDFs and pictures of a drawing (JPG, PNG, TIFF ...).
///
/// A picture has no lines or text to read, only pixels, so it is first traced into a drawing the PDF reader
/// understands, and then read exactly like a CAD PDF that was printed to PDF:
///   1. ink: dark or coloured pixels on the paper (uneven lighting of photos is evened out first)
///   2. words: read with OCR (cached, like printed PDFs); words of 3+ characters are wiped from the ink
///   3. thick ink (solid walls, filled dots) -> filled shapes (walls cut into pieces without holes, so the rooms
///      inside a wall ring stay open); thin ink -> centre lines, traced into polylines that end where lines meet,
///      like the separate lines of a CAD drawing (a wire ends at the symbol it feeds)
///   4. that traced drawing (one PDF page per picture page) is read by PdfReader: legend, symbols, rooms, areas
/// </summary>
public static unsafe class RasterReader
{
    private const int LongMin = 3000, LongMax = 7000;   // pixels: smaller pictures are enlarged (small words), bigger ones reduced (speed)
    private const double NominalDpi = 200;               // a picture without a trustworthy resolution: its "paper" size is unknown

    // ── small helpers over OpenCV images ──
    private static byte* B(Mat m) => (byte*)m.DataPointer;
    private static float* F(Mat m) => (float*)m.DataPointer;
    private static int* I(Mat m) => (int*)m.DataPointer;

    private static Mat Zeros(int h, int w, MatType t) => new(h, w, t, Scalar.All(0));

    private static Mat Kernel(int rows, int cols) => Mat.Ones(rows, cols, MatType.CV_8UC1).ToMat();

    private static Mat Ellipse(int w, int h) => Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(w, h));

    private static Mat RectK(int w, int h) => Cv2.GetStructuringElement(MorphShapes.Rect, new Size(w, h));

    private static Mat Dilate(Mat src, Mat kernel)
    {
        var dst = new Mat();
        Cv2.Dilate(src, dst, kernel);
        return dst;
    }

    private static Mat Morph(Mat src, MorphTypes op, Mat kernel)
    {
        var dst = new Mat();
        Cv2.MorphologyEx(src, dst, op, kernel);
        return dst;
    }

    private static Mat Roi(Mat m, int x0, int y0, int x1, int y1)
    {
        x0 = Math.Clamp(x0, 0, m.Cols); x1 = Math.Clamp(x1, 0, m.Cols);
        y0 = Math.Clamp(y0, 0, m.Rows); y1 = Math.Clamp(y1, 0, m.Rows);
        if (x1 <= x0 || y1 <= y0) return new Mat(Math.Max(0, y1 - y0), Math.Max(0, x1 - x0), m.Type(), Scalar.All(0));
        using var sub = new Mat(m, new CvRect(x0, y0, x1 - x0, y1 - y0));
        return sub.Clone();
    }

    private sealed class Components : IDisposable
    {
        public int N;
        public Mat Labels = new(), Stats = new(), Centroids = new();
        public int X(int i) => Stats.At<int>(i, 0);
        public int Y(int i) => Stats.At<int>(i, 1);
        public int W(int i) => Stats.At<int>(i, 2);
        public int H(int i) => Stats.At<int>(i, 3);
        public int Area(int i) => Stats.At<int>(i, 4);
        public double CX(int i) => Centroids.At<double>(i, 0);
        public double CY(int i) => Centroids.At<double>(i, 1);
        public void Dispose() { Labels.Dispose(); Stats.Dispose(); Centroids.Dispose(); }
    }

    private static Components CC(Mat img, PixelConnectivity conn = PixelConnectivity.Connectivity8)
    {
        var c = new Components();
        c.N = Cv2.ConnectedComponentsWithStats(img, c.Labels, c.Stats, c.Centroids, conn);
        return c;
    }

    /// <summary>numpy: np.bincount(values).argmax() - the most frequent value, the smallest on a tie</summary>
    private static int Mode(IEnumerable<int> values)
    {
        var counts = new Dictionary<int, int>();
        foreach (var v in values) counts[v] = (counts.TryGetValue(v, out var c) ? c : 0) + 1;
        if (counts.Count == 0) return 0;
        int best = -1, bc = -1;
        foreach (var (v, c) in counts.OrderBy(kv => kv.Key))
            if (c > bc) { best = v; bc = c; }
        return best;
    }

    // ── pictures ──
    private static (Mat Img, double S) Fit(Mat img)
    {
        int longSide = Math.Max(img.Rows, img.Cols);
        double s = 1.0;
        if (longSide < LongMin) s = (double)LongMin / longSide;
        else if (longSide > LongMax) s = (double)LongMax / longSide;
        if (Math.Abs(s - 1) > 0.02)
        {
            var dst = new Mat();
            Cv2.Resize(img, dst, new Size(0, 0), s, s, s > 1 ? InterpolationFlags.Cubic : InterpolationFlags.Area);
            img.Dispose();
            return (dst, s);
        }
        return (img, s);
    }

    /// <summary>(page number, RGB image, PDF points per pixel, is the paper size known, was it enlarged) for each page</summary>
    internal static List<(int Page, Mat Img, double Pt, bool Known, bool Enlarged)> Pictures(byte[] data, string kind, IList<int>? pages)
    {
        var out_ = new List<(int, Mat, double, bool, bool)>();
        if (kind == "scan")
        {
            using var doc = new PdfDoc(data);
            for (int i = 0; i < doc.PageCount; i++)
            {
                if (pages is { Count: > 0 } && !pages.Contains(i + 1)) continue;
                var page = doc[i];
                // the resolution the page was scanned at (its biggest picture), kept between 150 and 300 dpi
                double native = 200.0;
                var infos = page.ImageInfo().Where(x => x.Width != 0).ToList();
                if (infos.Count > 0)
                {
                    var big = Py.MaxBy(infos, x => (x.BBox.X1 - x.BBox.X0) * (x.BBox.Y1 - x.BBox.Y0));
                    double wIn = (big.BBox.X1 - big.BBox.X0) / 72;
                    if (wIn > 0) native = big.Width / wIn;
                }
                double dpi = Math.Min(Math.Max(native, 150), 300);
                dpi = Math.Max(Math.Min(dpi, LongMax * 72.0 / Math.Max(page.Rect.Width, page.Rect.Height)), 72);
                var (img, s) = Fit(page.RenderRgb(dpi));
                out_.Add((i + 1, img, 72 / (dpi * s), true, s > 1.02));
            }
            return out_;
        }
        double fileDpi = ImageFile.Dpi(data);
        bool known = fileDpi >= 100 && Py.RoundInt(fileDpi) is not (72 or 96);   // 72/96: screen defaults, not a real scan resolution
        var frames = ImageFile.Frames(data);
        for (int n = 1; n <= frames.Count; n++)                 // a multi-page TIFF: one page per frame
        {
            if (pages is { Count: > 0 } && !pages.Contains(n)) { frames[n - 1].Dispose(); continue; }
            var (img, s) = Fit(frames[n - 1]);
            out_.Add((n, img, 72 / ((known ? fileDpi : NominalDpi) * s), known, s > 1.02));
        }
        return out_;
    }

    // ── 1. ink ──
    private static readonly (int Upto, (double, double, double) Rgb)[] Hues =
    {
        (10, (1.0, 0.0, 0.0)), (22, (1.0, 0.5, 0.0)), (35, (0.8, 0.8, 0.0)), (85, (0.0, 0.6, 0.0)),
        (100, (0.0, 0.7, 0.7)), (130, (0.0, 0.0, 1.0)), (170, (0.8, 0.0, 0.8)), (181, (1.0, 0.0, 0.0)),
    };
    private static readonly (double, double, double)[] Colours = new[] { (0.0, 0.0, 0.0) }.Concat(Hues.Select(h => h.Rgb)).ToArray();

    /// <summary>the paper's colour at every pixel (photos are lit unevenly, paper is off-white): a closing of a small
    /// copy, which the thin lines and letters vanish from</summary>
    private static Mat Paper(Mat img)
    {
        using var small = new Mat();
        Cv2.Resize(img, small, new Size(0, 0), 0.25, 0.25, InterpolationFlags.Area);
        using var k = Ellipse(15, 15);
        using var paper = Morph(small, MorphTypes.Close, k);
        using var blurred = new Mat();
        Cv2.GaussianBlur(paper, blurred, new Size(0, 0), 3);
        var out_ = new Mat();
        Cv2.Resize(blurred, out_, new Size(img.Cols, img.Rows), 0, 0, InterpolationFlags.Linear);
        return out_;
    }

    /// <summary>the picture with white paper and the ink 3x darker: faint thin text (CAD stroke fonts) is found far
    /// more often by the text detector, and pale coloured lines show their colour clearly</summary>
    internal static Mat ForOcr(Mat img, Mat? paper = null)
    {
        var p = paper ?? Paper(img);
        using var p32 = new Mat(); using var i32 = new Mat();
        p.ConvertTo(p32, MatType.CV_32FC3);
        img.ConvertTo(i32, MatType.CV_32FC3);
        using var diff = new Mat();
        Cv2.Subtract(p32, i32, diff);
        using var clipped = new Mat();
        Cv2.Max(diff, Scalar.All(0), clipped);
        using var d = new Mat();
        Cv2.GaussianBlur(clipped, d, new Size(0, 0), 0.6);
        var out_ = new Mat(img.Rows, img.Cols, MatType.CV_8UC3);
        long total = (long)img.Rows * img.Cols * 3;
        float* src = F(d);
        byte* dst = B(out_);
        for (long i = 0; i < total; i++)
        {
            float v = 255f - 3f * src[i];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            dst[i] = (byte)v;
        }
        if (paper == null) p.Dispose();
        return out_;
    }

    /// <summary>OCR words of a picture, read several ways and the best kept: the contrast-stretched copy finds faint
    /// thin text, the plain picture reads small crisp text better (stretching thickens it), and a small picture that
    /// had to be enlarged is also read at a slightly smaller enlargement (OCR on tiny text is hit and miss)</summary>
    internal static List<OcrWord> ReadWords(Mat img, bool enlarged)
    {
        var variants = new List<(Mat, double, bool)> { (img, 1.0, false), (ForOcr(img), 1.0, true) };
        if (enlarged)
        {
            var r = new Mat();
            Cv2.Resize(img, r, new Size(0, 0), 0.75, 0.75, InterpolationFlags.Area);
            variants.Add((r, 0.75, true));
        }
        var words = new List<OcrWord>();
        foreach (var (v, s, own) in variants)
        {
            foreach (var w in Ocr.ImageWords(v))
                words.Add(new OcrWord(w.Text, w.Conf, w.X0 / s, w.Y0 / s, w.X1 / s, w.Y1 / s));
            if (own) v.Dispose();
        }
        // overlapping readings of the same place: the surest, longest one
        var sorted = Py.Sorted(words, w => -(w.Conf * w.Text.Trim().Length));
        var kept = new List<OcrWord>();
        foreach (var w in sorted)
        {
            double area = Math.Max((w.X1 - w.X0) * (w.Y1 - w.Y0), 1e-6);
            if (!kept.Any(k => Math.Max(0.0, Math.Min(w.X1, k.X1) - Math.Max(w.X0, k.X0)) * Math.Max(0.0, Math.Min(w.Y1, k.Y1) - Math.Max(w.Y0, k.Y0))
                               > 0.4 * Math.Min(area, (k.X1 - k.X0) * (k.Y1 - k.Y0))))
                kept.Add(w);
        }
        return kept;
    }

    /// <summary>numpy.percentile of a float32 array (linear; worked out in float32 as numpy 2 does)</summary>
    private static float PercentileF(float[] sorted, double q)
    {
        int n = sorted.Length;
        double virt = (n - 1) * (q / 100.0);
        int prev = Math.Clamp((int)Math.Floor(virt), 0, n - 1);
        int next = Math.Clamp(prev + 1, 0, n - 1);
        float g = (float)(virt - Math.Floor(virt));
        float a = sorted[prev], b = sorted[next];
        float diff = b - a;
        return g >= 0.5f ? b - diff * (1f - g) : a + diff * g;
    }

    /// <summary>(ink mask, colour index per pixel: 0 dark, else 1 + Hues index)</summary>
    internal static (Mat Mask, Mat Colour) Ink(Mat img)
    {
        int h = img.Rows, w = img.Cols;
        using var paper = Paper(img);
        // ink: darker than the paper in any colour channel (a pale green line is darker in red and blue), by more than
        // the paper's own grain (scanner noise, JPEG); a slight blur first calms the grain
        using var raw = new Mat(h, w, MatType.CV_32FC1);
        {
            byte* pp = B(paper), ip = B(img);
            float* rp = F(raw);
            long total = (long)h * w;
            for (long i = 0; i < total; i++)
            {
                int d0 = pp[3 * i] - ip[3 * i], d1 = pp[3 * i + 1] - ip[3 * i + 1], d2 = pp[3 * i + 2] - ip[3 * i + 2];
                rp[i] = Math.Max(d0, Math.Max(d1, d2));
            }
        }
        using var diff = new Mat();
        Cv2.GaussianBlur(raw, diff, new Size(0, 0), 0.8);
        float* dp = F(diff);
        var sample = new List<float>();
        for (int y = 0; y < h; y += 4)
            for (int x = 0; x < w; x += 4) sample.Add(dp[(long)y * w + x]);
        var sortedSample = sample.ToArray();
        Array.Sort(sortedSample);
        float pct = PercentileF(sortedSample, 60);
        var paperish = sample.Where(v => v < pct).ToArray();
        double level = paperish.Length > 0 ? Py.MedianF(paperish) : 0.0;
        double grain = paperish.Length > 0 ? 1.4826 * Py.MedianF(paperish.Select(p => Math.Abs(p - (float)level))) : 3.0;
        float strongT = (float)Math.Max(10.0, level + 6 * grain);
        float weakT = (float)Math.Max(6.0, level + 3.5 * grain);
        // faint stretches of a line (thinner or paler there) are kept when they belong to a long stroke or touch clear
        // ink - otherwise a light circle falls apart into arcs; grain at this level only makes small specks
        using var weak = new Mat(h, w, MatType.CV_8UC1);
        byte* wp = B(weak);
        long tot = (long)h * w;
        for (long i = 0; i < tot; i++) wp[i] = dp[i] > weakT ? (byte)1 : (byte)0;
        using var cc = CC(weak);
        var hasStrong = new bool[cc.N];
        int* lp = I(cc.Labels);
        for (long i = 0; i < tot; i++)
            if (dp[i] > strongT) hasStrong[lp[i]] = true;
        var keep = new bool[cc.N];
        for (int i = 1; i < cc.N; i++)
            keep[i] = (hasStrong[i] || Math.Max(cc.W(i), cc.H(i)) >= 15) && cc.Area(i) >= 4;     // (specks of dust and scanner noise out)
        var mask = new Mat(h, w, MatType.CV_8UC1);
        byte* mp = B(mask);
        for (long i = 0; i < tot; i++) mp[i] = keep[lp[i]] ? (byte)1 : (byte)0;
        // colour from the stretched picture: a pale green line is plainly green there
        using var stretched = ForOcr(img, paper);
        using var hsv = new Mat();
        Cv2.CvtColor(stretched, hsv, ColorConversionCodes.RGB2HSV);
        var colour = new Mat(h, w, MatType.CV_8UC1);
        byte* hp = B(hsv), cp = B(colour);
        for (long i = 0; i < tot; i++)
        {
            byte hh = hp[3 * i], ss = hp[3 * i + 1], vv = hp[3 * i + 2];
            if (ss > 70 && vv > 50)
            {
                int idx = 0;
                while (idx < Hues.Length && Hues[idx].Upto <= hh) idx++;      // searchsorted(..., side="right")
                cp[i] = (byte)(1 + Math.Min(idx, Hues.Length - 1));
            }
            else cp[i] = 0;
        }
        return (mask, colour);
    }

    /// <summary>ink pieces lying wholly inside a word of 3+ characters are its letters: wiped (short tags like "S1" or
    /// "AC" stay - they can be part of a symbol)</summary>
    internal static Mat WipeWords(Mat mask, IList<OcrWord> words)
    {
        var boxes = words.Where(w => Common.Norm(w.Text).Replace(" ", "").Length >= 3)
                         .Select(w => (X0: w.X0 - 2, Y0: w.Y0 - 2, X1: w.X1 + 2, Y1: w.Y1 + 2)).ToList();
        if (boxes.Count == 0) return mask;
        using var cc = CC(mask);
        if (cc.N <= 1) return mask;
        var tree = new Geo.Tree(boxes.Select(b => Geo.Env(b.X0, b.Y0, b.X1, b.Y1)));
        var drop = new bool[cc.N];
        for (int i = 1; i < cc.N; i++)
        {
            double x0 = cc.X(i), y0 = cc.Y(i), x1 = x0 + cc.W(i), y1 = y0 + cc.H(i);
            foreach (int j in tree.Query(Geo.Env(x0, y0, x1, y1)))
            {
                var b = boxes[j];
                if (b.X0 <= x0 && x1 <= b.X1 && b.Y0 <= y0 && y1 <= b.Y1) { drop[i] = true; break; }
            }
        }
        var out_ = mask.Clone();
        byte* op = B(out_);
        int* lp = I(cc.Labels);
        long tot = (long)mask.Rows * mask.Cols;
        for (long i = 0; i < tot; i++)
            if (drop[lp[i]]) op[i] = 0;
        mask.Dispose();
        return out_;
    }

    // ── 2. thin lines -> centre lines ──
    // 8 neighbours as bits: N, NE, E, SE, S, SW, W, NW
    private static readonly (byte[] Step1, byte[] Step2) Luts = MakeLuts();

    private static (byte[], byte[]) MakeLuts()
    {
        var step1 = new byte[256]; var step2 = new byte[256];
        for (int code = 0; code < 256; code++)
        {
            var p = Enumerable.Range(0, 8).Select(i => (code >> i) & 1).ToArray();   // N NE E SE S SW W NW
            int b = p.Sum();
            int a = Enumerable.Range(0, 8).Count(i => p[i] == 0 && p[(i + 1) % 8] == 1);
            int n = p[0], e = p[2], s = p[4], w = p[6];
            bool ok = 2 <= b && b <= 6 && a == 1;
            step1[code] = (byte)(ok && n * e * s == 0 && e * s * w == 0 ? 1 : 0);
            step2[code] = (byte)(ok && n * e * w == 0 && n * s * w == 0 ? 1 : 0);
        }
        return (step1, step2);
    }

    /// <summary>Zhang-Suen thinning: the centre line of every stroke, one pixel wide (only pixels next to ones just
    /// removed are looked at again - the same result as passes over the whole image)</summary>
    internal static byte[] Thin(Mat mask, out int width, out int height)
    {
        int h = mask.Rows, w = mask.Cols;
        int W = w + 2;
        var sk = new byte[(h + 2) * W];
        byte* mp = B(mask);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                sk[(y + 1) * W + x + 1] = mp[(long)y * w + x] > 0 ? (byte)1 : (byte)0;
        int[] offs = { -W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1 };    // bit order: N NE E SE S SW W NW
        int Code(int f)
        {
            int c = 0;
            for (int bit = 0; bit < 8; bit++) c |= sk[f + offs[bit]] << bit;
            return c;
        }
        var start = new List<int>();
        for (int f = 0; f < sk.Length; f++)
            if (sk[f] == 1 && Code(f) != 255) start.Add(f);        // pixels inside a stroke can't go before its edge does
        var todo = new List<int>[] { start, new List<int>(start) };
        var luts = new[] { Luts.Step1, Luts.Step2 };
        while (todo[0].Count > 0 || todo[1].Count > 0)
        {
            bool changed = false;
            for (int half = 0; half < 2; half++)
            {
                var flat = todo[half].Distinct().Where(f => sk[f] == 1).OrderBy(f => f).ToList();
                var lut = luts[half];
                var gone = flat.Where(f => lut[Code(f)] == 1).ToList();
                foreach (var f in gone) sk[f] = 0;
                // a pixel left alone keeps its verdict until something next to it goes
                var near = new List<int>();
                foreach (var f in gone)
                    foreach (var o in offs)
                        if (sk[f + o] == 1) near.Add(f + o);
                todo[half] = near;
                todo[1 - half] = todo[1 - half].Concat(near).ToList();
                changed |= gone.Count > 0;
            }
            if (!changed) break;
        }
        var out_ = new byte[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) out_[y * w + x] = sk[(y + 1) * W + x + 1];
        width = w;
        height = h;
        return out_;
    }

    /// <summary>centre lines -> polylines (x, y in pixels) that end at line ends and where lines meet; lines meeting at
    /// a junction all end at the same point (the junction's centre), so they still join up. With each polyline: the
    /// pixels it was traced from.</summary>
    internal static List<(Pt[] Pts, (int X, int Y)[] Px)> Trace(byte[] sk, int w, int h, double minSpur)
    {
        var ys = new List<int>(); var xs = new List<int>();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (sk[y * w + x] != 0) { ys.Add(y); xs.Add(x); }
        int n = ys.Count;
        if (n == 0) return new();
        var idx = new int[(h + 2) * (w + 2)];
        Array.Fill(idx, -1);
        int W2 = w + 2;
        for (int i = 0; i < n; i++) idx[(ys[i] + 1) * W2 + xs[i] + 1] = i;
        // neighbours: N E S W first, then NE SE SW NW
        (int dy, int dx)[] offs = { (-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1) };
        var nb = new int[n, 8];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 8; k++) nb[i, k] = idx[(ys[i] + 1 + offs[k].dy) * W2 + xs[i] + 1 + offs[k].dx];
        // a diagonal step is dropped where a pixel beside it already links the two (stair steps of a sloping line
        // would otherwise look like junctions)
        var pairs = new[] { (0, 1), (1, 2), (2, 3), (3, 0) };
        for (int i = 0; i < n; i++)
        {
            var orth = new[] { nb[i, 0] >= 0, nb[i, 1] >= 0, nb[i, 2] >= 0, nb[i, 3] >= 0 };
            for (int k = 4; k < 8; k++)
            {
                var (a, b) = pairs[k - 4];
                if (orth[a] || orth[b]) nb[i, k] = -1;
            }
        }
        var deg = new int[n];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 8; k++) if (nb[i, k] >= 0) deg[i]++;
        var node = deg.Select(d => d != 2).ToArray();     // line ends (1), junctions (3+), dots (0)
        // junction pixels next to each other are one junction: lines meeting there all end at its centre
        using var jmask = Zeros(h, w, MatType.CV_8UC1);
        byte* jm = B(jmask);
        for (int i = 0; i < n; i++) if (deg[i] >= 3) jm[(long)ys[i] * w + xs[i]] = 1;
        using var jcc = CC(jmask);
        int* jl = I(jcc.Labels);
        var jof = new int[n];
        for (int i = 0; i < n; i++) jof[i] = jl[(long)ys[i] * w + xs[i]];
        var nbl = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            nbl[i] = new List<int>(4);
            for (int k = 0; k < 8; k++) if (nb[i, k] >= 0) nbl[i].Add(nb[i, k]);
        }
        var seen = new bool[n];
        var used = new HashSet<(int, int)>();
        var paths = new List<(List<int> Path, bool Closed)>();
        for (int s = 0; s < n; s++)                 // every branch from every end and junction, to the next one
        {
            if (!node[s]) continue;
            foreach (int t in nbl[s])
            {
                if (node[t])
                {
                    var key = s < t ? (s, t) : (t, s);
                    if (used.Add(key)) paths.Add((new List<int> { s, t }, false));
                    continue;
                }
                if (seen[t]) continue;
                var path = new List<int> { s, t };
                int prev = s, cur = t;
                while (!node[cur])
                {
                    seen[cur] = true;
                    int a = nbl[cur][0], b = nbl[cur][1];
                    int nxt = a == prev ? b : a;
                    prev = cur;
                    cur = nxt;
                    path.Add(cur);
                }
                paths.Add((path, false));
            }
        }
        for (int s = 0; s < n; s++)                 // what is left: rings (the circle of a symbol)
        {
            if (node[s] || seen[s]) continue;
            var path = new List<int> { s };
            int prev = -1, cur = s;
            while (true)
            {
                seen[cur] = true;
                int a = nbl[cur][0], b = nbl[cur][1];
                int nxt = a == prev ? b : a;
                if (nxt == s || seen[nxt]) break;
                path.Add(nxt);
                prev = cur;
                cur = nxt;
            }
            paths.Add((path, true));
        }
        var out_ = new List<(Pt[], (int, int)[])>();
        foreach (var (p, closed) in paths)
        {
            var px = p.Select(i => (xs[i], ys[i])).ToArray();
            var pts = px.Select(q => new Pt(q.Item1, q.Item2)).ToArray();
            if (!closed)
            {
                int j0 = jof[p[0]], j1 = jof[p[^1]];
                if (j0 != 0 && j0 == j1 && p.Count <= 4) continue;           // inside one junction
                int free = (deg[p[0]] == 1 ? 1 : 0) + (deg[p[^1]] == 1 ? 1 : 0);
                if (free == 1 && (j0 != 0 || j1 != 0) && p.Count < minSpur) continue;   // a whisker the thinning left at a corner or a line end
                if (j0 != 0) pts[0] = new Pt(jcc.CX(j0), jcc.CY(j0));
                if (j1 != 0) pts[^1] = new Pt(jcc.CX(j1), jcc.CY(j1));
            }
            if (pts.Length < 2) continue;
            var approx = Cv2.ApproxPolyDP(pts.Select(q => new Point2f((float)q.X, (float)q.Y)).ToArray(), 0.9, closed)
                            .Select(q => new Pt(q.X, q.Y)).ToList();
            if (closed) approx.Add(approx[0]);
            else
            {
                approx[0] = pts[0];                 // the junction points exactly, not rounded by the simplification
                approx[^1] = pts[^1];
            }
            if (approx.Count >= 2) out_.Add((approx.ToArray(), px));
        }
        return out_;
    }

    // ── 3. thick ink -> filled shapes ──
    /// <summary>a wall ring has the rooms as holes; filled shapes can't (each outline becomes solid): cut it through its
    /// holes into pieces without any</summary>
    private static List<Polygon> Holeless(Polygon poly)
    {
        var out_ = new List<Polygon>();
        var todo = new List<Polygon> { poly };
        while (todo.Count > 0)
        {
            var p = todo[^1];
            todo.RemoveAt(todo.Count - 1);
            if (p.IsEmpty || p.Area < 1) continue;
            if (p.NumInteriorRings == 0)
            {
                out_.Add(p);
                continue;
            }
            var (x0, y0, x1, y1) = Geo.Bounds(p);
            double hx = Geo.Centroid(Geo.F.CreatePolygon((LinearRing)p.GetInteriorRingN(0))).X;
            foreach (var half in new[] { Geo.Box(x0 - 1, y0 - 1, hx, y1 + 1), Geo.Box(hx, y0 - 1, x1 + 1, y1 + 1) })
            {
                var part = Geo.Intersection(p, half);
                todo.AddRange(Geo.Parts(part).OfType<Polygon>());
            }
        }
        return out_;
    }

    /// <summary>Touching filled dots (switches stacked on a wall) print as one blob; a CAD drawing has each dot apart.
    /// A blob whose distance map has several round peaks, each owning about a circle's worth of the blob, is cut
    /// between them (wall strips have a ridge, not round peaks, and are left alone).</summary>
    internal static Mat SplitDots(Mat thick, int maxSide)
    {
        using var cc = CC(thick);
        var out_ = thick.Clone();
        int W = thick.Cols;
        int* lab = I(cc.Labels);
        for (int i = 1; i < cc.N; i++)
        {
            int x = cc.X(i), y = cc.Y(i), w = cc.W(i), h = cc.H(i), area = cc.Area(i);
            if (Math.Max(w, h) > maxSide || area < 20) continue;
            using var sub = Zeros(h, w, MatType.CV_8UC1);
            byte* sp = B(sub);
            for (int yy = 0; yy < h; yy++)
                for (int xx = 0; xx < w; xx++)
                    if (lab[(long)(y + yy) * W + x + xx] == i) sp[yy * w + xx] = 1;
            using var padded = new Mat();
            Cv2.CopyMakeBorder(sub, padded, 1, 1, 1, 1, BorderTypes.Constant, Scalar.All(0));
            using var dfull = new Mat();
            Cv2.DistanceTransform(padded, dfull, DistanceTypes.L2, DistanceTransformMasks.Mask5);
            using var d = Roi(dfull, 1, 1, w + 1, h + 1);
            Cv2.MinMaxLoc(d, out double _, out double rmax);
            float r = (float)rmax;
            if (r < 2) continue;
            int k = Math.Max(3, (int)(1.4 * r) | 1);
            using var ek = Ellipse(k, k);
            using var dd = Dilate(d, ek);
            using var peaks = Zeros(h, w, MatType.CV_8UC1);
            float* dp = F(d); float* ddp = F(dd);
            byte* pk = B(peaks);
            float thr = (float)(0.6 * r);
            for (int q = 0; q < h * w; q++)
                pk[q] = dp[q] >= ddp[q] - 1e-3f && dp[q] >= thr ? (byte)1 : (byte)0;
            using var pcc = CC(peaks);
            if (pcc.N - 1 < 2) continue;
            int m = pcc.N - 1;
            var cent = Enumerable.Range(1, m).Select(c => (X: pcc.CX(c), Y: pcc.CY(c))).ToArray();
            var radius = cent.Select(c => dp[Math.Clamp((int)Math.Round(c.Y, MidpointRounding.ToEven), 0, h - 1) * w
                                             + Math.Clamp((int)Math.Round(c.X, MidpointRounding.ToEven), 0, w - 1)]).ToArray();
            var cell = new int[h * w];
            Array.Fill(cell, -1);
            var cells = new long[m];
            for (int yy = 0; yy < h; yy++)
                for (int xx = 0; xx < w; xx++)
                {
                    if (sp[yy * w + xx] == 0) continue;
                    int best = 0;
                    double bd = double.PositiveInfinity;
                    for (int c = 0; c < m; c++)
                    {
                        double ddx = xx - cent[c].X, ddy = yy - cent[c].Y;
                        double dist = ddx * ddx + ddy * ddy;
                        if (dist < bd) { bd = dist; best = c; }
                    }
                    cell[yy * w + xx] = best;
                    cells[best]++;
                }
            bool notRound = false;
            for (int c = 0; c < m; c++)
            {
                float rr = Math.Max(radius[c], 1f);
                if (cells[c] > (double)((float)(1.9 * Math.PI) * (rr * rr))) { notRound = true; break; }
            }
            if (notRound) continue;                         // not round dots
            // a one-pixel cut wherever two dots' pixels meet
            using var cut = Zeros(h, w, MatType.CV_8UC1);
            byte* cp = B(cut);
            for (int yy = 0; yy < h; yy++)
                for (int xx = 0; xx < w; xx++)
                {
                    int c0 = cell[yy * w + xx];
                    if (c0 < 0) continue;
                    if (xx > 0 && cell[yy * w + xx - 1] >= 0 && cell[yy * w + xx - 1] != c0) cp[yy * w + xx] = 1;
                    if (yy > 0 && cell[(yy - 1) * w + xx] >= 0 && cell[(yy - 1) * w + xx] != c0) cp[yy * w + xx] = 1;
                }
            using var k3 = Kernel(3, 3);
            using var cutD = Dilate(cut, k3);                // no corner left touching
            byte* cdp = B(cutD);
            byte* op = B(out_);
            for (int yy = 0; yy < h; yy++)
                for (int xx = 0; xx < w; xx++)
                    if (cdp[yy * w + xx] > 0) op[(long)(y + yy) * W + x + xx] = 0;
        }
        return out_;
    }

    /// <summary>each piece of thick ink as outlines without holes (pixels)</summary>
    private static List<List<Pt[]>> Solids(Mat thick)
    {
        Cv2.FindContours(thick, out CvPoint[][] contours, out HierarchyIndex[] hier, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);
        var out_ = new List<List<Pt[]>>();
        if (hier == null || hier.Length == 0) return out_;
        for (int i = 0; i < contours.Length; i++)
        {
            var c = contours[i];
            if (hier[i].Parent != -1 || c.Length < 3) continue;      // a hole: handled with its outer outline
            var holes = new List<IReadOnlyList<Pt>>();
            int k = hier[i].Child;
            while (k != -1)
            {
                if (contours[k].Length >= 3) holes.Add(contours[k].Select(p => new Pt(p.X, p.Y)).ToList());
                k = hier[k].Next;
            }
            Geometry poly = Geo.Polygon(c.Select(p => new Pt(p.X, p.Y)).ToList(), holes);
            if (!Geo.IsValid(poly)) poly = Geo.Buffer(poly, 0);
            var pieces = new List<Pt[]>();
            foreach (var g in Geo.Parts(poly).OfType<Polygon>())
            {
                var simplified = Geo.Simplify(g, 0.7);
                foreach (var sg in Geo.Parts(simplified).OfType<Polygon>())
                    foreach (var piece in Holeless(sg))
                    {
                        var pts = piece.ExteriorRing.Coordinates.Select(q => new Pt(q.X, q.Y)).ToArray();
                        if (pts.Length >= 4) pieces.Add(pts);
                    }
            }
            if (pieces.Count > 0) out_.Add(pieces);
        }
        return out_;
    }

    /// <summary>A pen weight per line: widths measured from pixels wander along a line, so close widths (within about a
    /// third) are one weight - walls then share one weight, as in the CAD drawing (rooms are traced by weight)</summary>
    private static List<(Pt[] Pts, double Width, (double, double, double) Col)> LineWeights(List<(Pt[] Pts, double Width, (double, double, double) Col, int Len)> lines)
    {
        if (lines.Count == 0) return new();
        var widths = lines.Select(l => l.Width).ToArray();
        var length = lines.Select(l => (double)l.Len).ToArray();
        var order = Py.Sorted(Enumerable.Range(0, lines.Count), i => widths[i]);
        var groups = new List<List<int>> { new() { order[0] } };
        foreach (var i in order.Skip(1))
        {
            if (widths[i] <= 1.35 * widths[groups[^1][0]]) groups[^1].Add(i);
            else groups.Add(new() { i });
        }
        var weight = new double[lines.Count];
        foreach (var g in groups)
        {
            var o = Py.Sorted(g, i => widths[i]);
            var cum = new double[o.Count];
            double acc = 0;
            for (int q = 0; q < o.Count; q++) { acc += length[o[q]]; cum[q] = acc; }
            double half = cum[^1] / 2;
            int pos = 0;
            while (pos < cum.Length && cum[pos] < half) pos++;        // searchsorted (left)
            double wv = widths[o[Math.Min(pos, o.Count - 1)]];        // the length-weighted median
            foreach (var i in g) weight[i] = wv;
        }
        return lines.Select((l, i) => (l.Pts, Py.Round(weight[i], 2), l.Col)).ToList();
    }

    // ── 4. the traced drawing ──
    internal sealed class Traced : IDisposable
    {
        public double LinePx, SolidPx;
        public Mat Mask = null!, SymbolInk = null!, Colour = null!, Walls = null!;
        public (int X0, int Y0, int X1, int Y1)? Footprint;
        public void Dispose() { Mask.Dispose(); SymbolInk.Dispose(); Colour.Dispose(); Walls.Dispose(); }
    }

    /// <summary>(filled shapes [(outlines, colour)], lines [(points, width px, colour)], what was measured) of one picture</summary>
    internal static (List<(List<Pt[]> Pieces, (double, double, double) Col)> Filled, List<(Pt[] Pts, double Width, (double, double, double) Col)> Lines, Traced St)
        Vectorise(Mat img, IList<OcrWord> words)
    {
        var (mask0, colour) = Ink(img);
        var mask = WipeWords(mask0, words);
        int h = mask.Rows, wd = mask.Cols;
        long tot = (long)h * wd;
        using var dist = new Mat();
        Cv2.DistanceTransform(mask, dist, DistanceTypes.L2, DistanceTransformMasks.Mask3);
        using var k3 = Kernel(3, 3);
        using var distD = Dilate(dist, k3);
        float* dp = F(dist); float* ddp = F(distD);
        byte* mp = B(mask);
        var ridge = new List<float>();
        for (long i = 0; i < tot; i++)
            if (mp[i] > 0 && dp[i] >= ddp[i] - 1e-6f) ridge.Add(dp[i]);
        double w = ridge.Count > 0 ? Py.MedianF(ridge) * 2.0 : 2.0;      // usual line width, pixels
        w = Math.Max(w, 1.0);
        int k = Py.RoundInt(Math.Max(2.5 * w, w + 3));
        using var ek = Ellipse(k, k);
        var thick = Morph(mask, MorphTypes.Open, ek);
        var thinInk = mask.Clone();
        using (var thickD = Dilate(thick, k3))
        {
            byte* tdp = B(thickD); byte* tip = B(thinInk);
            for (long i = 0; i < tot; i++) if (tdp[i] > 0) tip[i] = 0;
        }
        // the sheet frame and border lines are often drawn heavy: thick ink running (nearly) edge to edge is not a wall
        using var cc = CC(thick);
        var frame = new bool[cc.N];
        for (int i = 1; i < cc.N; i++) frame[i] = cc.W(i) > 0.75 * wd || cc.H(i) > 0.75 * h;
        int* lab = I(cc.Labels);
        if (frame.Any(f => f))
        {
            byte* tp = B(thick);
            for (long i = 0; i < tot; i++) if (frame[lab[i]]) tp[i] = 0;
        }
        // the building: the big masses of solid wall (symbols are only looked for inside it - not in the title block)
        var big = Enumerable.Range(1, Math.Max(0, cc.N - 1)).Where(i => !frame[i] && Math.Max(cc.W(i), cc.H(i)) >= 0.1 * Math.Min(h, wd)).ToList();
        (int, int, int, int)? footprint = null;
        if (big.Count > 0)
        {
            int top = big.Max(i => cc.Area(i));
            var boxes = big.Where(i => cc.Area(i) >= 0.15 * top).ToList();
            int x0 = boxes.Min(i => cc.X(i)), y0 = boxes.Min(i => cc.Y(i));
            int x1 = boxes.Max(i => cc.X(i) + cc.W(i)), y1 = boxes.Max(i => cc.Y(i) + cc.H(i));
            int pad = (int)(0.02 * Math.Max(h, wd));
            footprint = (Math.Max(0, x0 - pad), Math.Max(0, y0 - pad), Math.Min(wd, x1 + pad), Math.Min(h, y1 + pad));
        }
        // walls (big solid pieces) are left out when symbols are looked for: a solid wall "matches" a filled symbol
        var walls = Zeros(h, wd, MatType.CV_8UC1);
        var bigParts = Enumerable.Range(1, Math.Max(0, cc.N - 1)).Where(i => !frame[i] && Math.Max(cc.W(i), cc.H(i)) > 8 * k).ToHashSet();
        byte* wlp = B(walls);
        byte* clp = B(colour);
        if (bigParts.Count > 0)
            for (long i = 0; i < tot; i++) if (bigParts.Contains(lab[i])) wlp[i] = 1;
        // coloured ink is never wall: switch dots drawn against a wall face stay symbols
        for (long i = 0; i < tot; i++) if (clp[i] > 0) wlp[i] = 0;
        var symbolInk = mask.Clone();
        int edge = Math.Max(5, k / 2 + 3) | 1;                 // (and the wall's soft edges with it)
        using (var ke = Kernel(edge, edge))
        using (var wallsD = Dilate(walls, ke))
        {
            byte* wdp = B(wallsD); byte* sip = B(symbolInk);
            for (long i = 0; i < tot; i++) if (wdp[i] > 0 && clp[i] == 0) sip[i] = 0;
        }
        var split = SplitDots(thick, (int)(12 * k));
        thick.Dispose();
        var sk = Thin(thinInk, out int skW, out int skH);
        thinInk.Dispose();
        var filled = new List<(List<Pt[]>, (double, double, double))>();
        foreach (var pieces in Solids(split))
        {
            var cols = pieces.Select(p => Geo.InteriorPoint(Geo.Polygon(p))).Select(q =>
                clp[(long)Math.Clamp((int)q.Y, 0, h - 1) * wd + Math.Clamp((int)q.X, 0, wd - 1)]).Select(v => (int)v);
            filled.Add((pieces, Colours[Mode(cols)]));
        }
        split.Dispose();
        var lines = new List<(Pt[], double, (double, double, double), int)>();
        foreach (var (pts, px) in Trace(sk, skW, skH, 2 * w + 2))
        {
            double width = Math.Max(1.0, Py.MedianF(px.Select(q => dp[(long)q.Y * wd + q.X])) * 2.0);
            lines.Add((pts, width, Colours[Mode(px.Select(q => (int)clp[(long)q.Y * wd + q.X]))], px.Length));
        }
        var weighted = LineWeights(lines);
        return (filled, weighted, new Traced { LinePx = w, SolidPx = k, Mask = mask, SymbolInk = symbolInk, Colour = colour, Footprint = footprint, Walls = walls });
    }

    // ── 5. symbols: each legend symbol, cut from the picture, looked for over the plan ──
    // chamfer matching (pixels at the working size, symbol ~32 px): how far the symbol's strokes are from the plan's,
    // on average, and the plan's strokes there from the symbol's
    private const float ForwardMax = 2.0f, BackwardMax = 3.0f;
    private const float CoverMin = 0.75f;                  // share of the symbol's ink found on the plan (with a pixel of slack)
    private const double SymbolPx = 20.0;                  // symbols are found at about this size ...
    private const double FinePx = 48.0;                    // ... and told apart at this size
    private const double MatchMin = 0.38;                  // score (1 / (1 + cost)) of a match
    private const double CoarseMin = 0.30;                 // the quick search's score worth a closer look
    private const int MaxPlaces = 2000;                    // places looked at closely per page (strongest first)
    private const float Clip = 6.0f;

    /// <summary>the outline of ink (a thin line is its own outline; a filled dot becomes a ring)</summary>
    private static Mat Edges(Mat b)
    {
        using var k3 = Kernel(3, 3);
        using var er = new Mat();
        Cv2.Erode(b, er, k3);
        var out_ = new Mat();
        Cv2.Subtract(b, er, out_);
        return out_;
    }

    internal sealed class Template : IDisposable
    {
        public string Kind = "", Desc = "";
        public Mat Ink = null!;
        public int Colour;
        public int Size;
        public void Dispose() => Ink.Dispose();
    }

    /// <summary>the symbol(s) drawn in each legend row, as small ink pictures</summary>
    internal static List<Template> Templates(Mat mask, Mat colour, List<(string Kind, string Desc, PRect Strip)> rows, double pt)
    {
        var out_ = new List<Template>();
        int H = mask.Rows, W = mask.Cols;
        foreach (var (kind, desc, strip) in rows)
        {
            int x0 = Math.Max(0, (int)(strip.X0 / pt)), y0 = Math.Max(0, (int)(strip.Y0 / pt));
            int x1 = Math.Min(W, (int)(strip.X1 / pt)), y1 = Math.Min(H, (int)(strip.Y1 / pt));
            if (x1 - x0 < 5 || y1 - y0 < 5) continue;
            using var cell = Roi(mask, x0, y0, x1, y1);
            int h = cell.Rows, w = cell.Cols;
            // table borders: straight lines running (nearly) the whole cell - a scan is often a little tilted, so the
            // line is thickened across first (a tilted line steps by a pixel or two over the cell)
            using (var k51 = Kernel(5, 1)) using (var k15 = Kernel(1, 5))
            using (var rh = RectK(Math.Max(15, (int)(0.7 * w)), 1)) using (var rv = RectK(1, Math.Max(15, (int)(0.8 * h))))
            using (var dh = Dilate(cell, k51)) using (var dv = Dilate(cell, k15))
            using (var hor = Morph(dh, MorphTypes.Open, rh)) using (var ver = Morph(dv, MorphTypes.Open, rv))
            using (var both = new Mat())
            using (var k3 = Kernel(3, 3))
            {
                Cv2.BitwiseOr(hor, ver, both);
                using var bd = Dilate(both, k3);
                byte* bp = B(bd); byte* cp = B(cell);
                for (int q = 0; q < h * w; q++) if (bp[q] > 0) cp[q] = 0;
            }
            using var cc = CC(cell);
            int biggest = Enumerable.Range(1, Math.Max(0, cc.N - 1)).Select(i => cc.Area(i)).DefaultIfEmpty(0).Max();
            var comps = Enumerable.Range(1, Math.Max(0, cc.N - 1)).Where(i => cc.Area(i) >= Math.Max(6, 0.03 * biggest))
                                  .Select(i => (X: cc.X(i), Y: cc.Y(i), W: cc.W(i), H: cc.H(i))).ToList();
            if (comps.Count == 0) continue;
            // a row can show several symbols side by side (normal / smart / dimmer switch): split at wide gaps
            comps = Py.Sorted(comps, c => c.X);
            var groups = new List<List<(int X, int Y, int W, int H)>> { new() { comps[0] } };
            double gap = 0.25 * h;
            foreach (var c in comps.Skip(1))
            {
                int right = groups[^1].Max(g => g.X + g.W);
                if (c.X - right < gap) groups[^1].Add(c);
                else groups.Add(new() { c });
            }
            var boxes = new List<(int, int, int, int)>();
            foreach (var g in groups)
            {
                int gx0 = g.Min(c => c.X), gy0 = g.Min(c => c.Y), gx1 = g.Max(c => c.X + c.W), gy1 = g.Max(c => c.Y + c.H);
                boxes.AddRange(SplitRepeats(cell, (gx0, gy0, gx1, gy1)));
            }
            var sizes = boxes.Select(b => Math.Max(b.Item3 - b.Item1, b.Item4 - b.Item2)).ToList();
            int topSize = sizes.Count > 0 ? sizes.Max() : 0;
            using var colCell = Roi(colour, x0, y0, x1, y1);
            for (int bi = 0; bi < boxes.Count; bi++)
            {
                var (gx0, gy0, gx1, gy1) = boxes[bi];
                int size = sizes[bi];
                if (size < 0.35 * topSize || size < 0.15 * h) continue;                     // a speck or a tick, not a symbol
                if (Math.Abs((gy0 + gy1) / 2.0 - h / 2.0) > 0.3 * h || gy0 <= 0 || gy1 >= h) continue;   // not on the row's line, or cut by the cell
                if (Math.Min(gx1 - gx0, gy1 - gy0) < 0.25 * size) continue;                // a sliver (a piece of a border line)
                var ink = Roi(cell, Math.Max(0, gx0 - 2), Math.Max(0, gy0 - 2), gx1 + 2, gy1 + 2);
                using var cols = Roi(colCell, Math.Max(0, gx0 - 2), Math.Max(0, gy0 - 2), gx1 + 2, gy1 + 2);
                byte* ip = B(ink); byte* cp = B(cols);
                var inkCols = new List<int>();
                for (int q = 0; q < ink.Rows * ink.Cols; q++) if (ip[q] > 0) inkCols.Add(cp[q]);
                out_.Add(new Template { Kind = kind, Desc = desc, Ink = ink, Colour = Mode(inkCols), Size = size });
            }
        }
        return out_;
    }

    /// <summary>A row showing the same symbol in a few variants side by side (a dot with differently drawn ticks:
    /// normal / smart / dimmer switch) is cut halfway between its dots, one symbol each</summary>
    private static List<(int, int, int, int)> SplitRepeats(Mat cell, (int X0, int Y0, int X1, int Y1) b)
    {
        var (x0, y0, x1, y1) = b;
        int w = x1 - x0, h = y1 - y0;
        if (w < 1.5 * h) return new() { b };
        using var part = Roi(cell, x0, y0, x1, y1);
        int k = Math.Max(3, (int)(0.15 * h) | 1);
        using var ek = Ellipse(k, k);
        using var dots = Morph(part, MorphTypes.Open, ek);
        using var cc = CC(dots);
        if (cc.N - 1 < 2) return new() { b };
        var areas = Enumerable.Range(1, cc.N - 1).Select(i => (double)cc.Area(i)).ToList();
        double med = Py.Median(areas);
        var xs = Enumerable.Range(1, cc.N - 1).Where(i => cc.Area(i) >= 0.5 * med).Select(i => cc.CX(i)).OrderBy(v => v).ToList();
        if (xs.Count < 2) return new() { b };
        var cuts = new List<int> { 0 };
        for (int i = 0; i + 1 < xs.Count; i++) cuts.Add((int)((xs[i] + xs[i + 1]) / 2));
        cuts.Add(w);
        var out_ = new List<(int, int, int, int)>();
        byte* pp = B(part);
        int pw = part.Cols, ph = part.Rows;
        for (int ci = 0; ci + 1 < cuts.Count; ci++)
        {
            int a = Math.Max(0, cuts[ci]), c = Math.Min(pw, cuts[ci + 1]);
            int c0 = -1, c1 = -1, r0 = -1, r1 = -1;
            for (int xx = a; xx < c; xx++)
                for (int yy = 0; yy < ph; yy++)
                    if (pp[yy * pw + xx] != 0)
                    {
                        if (c0 < 0) c0 = xx;
                        c1 = xx;
                        if (r0 < 0 || yy < r0) r0 = yy;
                        if (yy > r1) r1 = yy;
                    }
            if (c0 >= 0 && r0 >= 0) out_.Add((x0 + c0, y0 + r0, x0 + c1 + 1, y0 + r1 + 1));
        }
        return out_.Count > 0 ? out_ : new() { b };
    }

    /// <summary>the plan's ink at one working size, ready for chamfer matching</summary>
    private sealed class PlanIndex : IDisposable
    {
        public double Fz;
        public int Rows, Cols;
        public Mat Reach, Loose, EdgesF, Near, LooseSum, EdgesSum;

        public PlanIndex(Mat plan, double f, double symbolPx)
        {
            Fz = f;
            using var pf = new Mat();
            plan.ConvertTo(pf, MatType.CV_32FC1);
            using var rs = new Mat();
            Cv2.Resize(pf, rs, new Size(0, 0), f, f, InterpolationFlags.Area);
            using var small = new Mat(rs.Rows, rs.Cols, MatType.CV_8UC1);
            float* rp = F(rs); byte* sp = B(small);
            for (int i = 0; i < rs.Rows * rs.Cols; i++) sp[i] = rp[i] > 0.2f ? (byte)1 : (byte)0;
            Rows = small.Rows; Cols = small.Cols;
            using var edges = Edges(small);
            using var inv = new Mat();
            Cv2.Subtract(Scalar.All(1), edges, inv);
            using var dt = new Mat();
            Cv2.DistanceTransform(inv, dt, DistanceTypes.L2, DistanceTransformMasks.Mask3);
            Reach = new Mat();
            Cv2.Min(dt, Clip, Reach);
            // long straight lines (walls, wires, a tile grid) don't count as extra strokes in a symbol's window
            int n = (int)(1.6 * symbolPx);
            using var rh = RectK(n, 1); using var rv = RectK(1, n);
            using var lh = Morph(small, MorphTypes.Open, rh); using var lv = Morph(small, MorphTypes.Open, rv);
            using var longL = new Mat();
            Cv2.BitwiseOr(lh, lv, longL);
            using var k3 = Kernel(3, 3);
            using var longD = Dilate(longL, k3);
            Loose = new Mat(Rows, Cols, MatType.CV_32FC1);
            EdgesF = new Mat(Rows, Cols, MatType.CV_32FC1);
            byte* ep = B(edges); byte* ldp = B(longD);
            float* lp = F(Loose); float* efp = F(EdgesF);
            for (int i = 0; i < Rows * Cols; i++)
            {
                lp[i] = (ep[i] & (ldp[i] == 0 ? 1 : 0)) != 0 ? 1f : 0f;
                efp[i] = ep[i];
            }
            using var nearB = Dilate(small, k3);
            Near = new Mat();
            nearB.ConvertTo(Near, MatType.CV_32FC1);
            LooseSum = new Mat(); EdgesSum = new Mat();
            Cv2.Integral(Loose, LooseSum, MatType.CV_64F);
            Cv2.Integral(EdgesF, EdgesSum, MatType.CV_64F);
        }

        public void Dispose() { Reach.Dispose(); Loose.Dispose(); EdgesF.Dispose(); Near.Dispose(); LooseSum.Dispose(); EdgesSum.Dispose(); }
    }

    /// <summary>the sum over every h x w window (top-left corners, like matchTemplate) from an integral image</summary>
    private static float[] WindowSums(Mat ii, int y0, int x0, int h, int w, int rows, int cols)
    {
        var out_ = new float[rows * cols];
        int stride = ii.Cols;
        double* p = (double*)ii.DataPointer;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                double a = p[(y0 + r) * stride + x0 + c], b = p[(y0 + r) * stride + x0 + w + c];
                double cc = p[(y0 + h + r) * stride + x0 + c], d = p[(y0 + h + r) * stride + x0 + w + c];
                out_[r * cols + c] = (float)(d - b - cc + a);
            }
        return out_;
    }

    private sealed class Shape : IDisposable
    {
        public Mat E = null!, Reach = null!, Ink = null!;
        public float Own, Solid;
        public int M;
        public void Dispose() { E.Dispose(); Reach.Dispose(); Ink.Dispose(); }
    }

    private static Mat Rot90(Mat m, int k)
    {
        var dst = new Mat();
        switch (k % 4)
        {
            case 0: m.CopyTo(dst); break;
            case 1: Cv2.Rotate(m, dst, RotateFlags.Rotate90Counterclockwise); break;
            case 2: Cv2.Rotate(m, dst, RotateFlags.Rotate180); break;
            default: Cv2.Rotate(m, dst, RotateFlags.Rotate90Clockwise); break;
        }
        return dst;
    }

    /// <summary>a legend symbol at working size f, in its four quarter turns</summary>
    private static List<Shape?> Shapes(Template t, double f)
    {
        using var inkF = new Mat();
        t.Ink.ConvertTo(inkF, MatType.CV_32FC1);
        using var rs = new Mat();
        Cv2.Resize(inkF, rs, new Size(0, 0), f, f, InterpolationFlags.Area);
        using var tb0 = new Mat(rs.Rows, rs.Cols, MatType.CV_8UC1);
        float* rp = F(rs); byte* tp = B(tb0);
        for (int i = 0; i < rs.Rows * rs.Cols; i++) tp[i] = rp[i] > 0.2f ? (byte)1 : (byte)0;
        // a margin around the symbol: strokes just outside it count against it (a small symbol sitting inside a bigger
        // one - a switch's dot and tick inside a light's circle - is not that small symbol)
        int m = Math.Max(3, (int)(0.25 * Math.Max(tb0.Rows, tb0.Cols)));
        using var tb = new Mat();
        Cv2.CopyMakeBorder(tb0, tb, m, m, m, m, BorderTypes.Constant, Scalar.All(0));
        using var te = Edges(tb);
        double teSum = Cv2.Sum(te).Val0;
        if (teSum < 8) return new();
        using var inv = new Mat();
        Cv2.Subtract(Scalar.All(1), te, inv);
        using var dt = new Mat();
        Cv2.DistanceTransform(inv, dt, DistanceTypes.L2, DistanceTransformMasks.Mask3);
        using var reach = new Mat();
        Cv2.Min(dt, Clip, reach);
        double solid = Cv2.Sum(tb).Val0;
        var out_ = new List<Shape?>();
        for (int k = 0; k < 4; k++)
        {
            var r = Rot90(te, k);
            bool same = false;
            foreach (var o in out_)
            {
                if (o == null) continue;           // (a turn already found the same: nothing to compare)
                if (o.E.Rows == r.Rows && o.E.Cols == r.Cols)
                {
                    float* oe = F(o.E); byte* rr = B(r);
                    long both = 0;
                    for (int i = 0; i < r.Rows * r.Cols; i++) if (oe[i] > 0 && rr[i] > 0) both++;
                    if (both > 0.9 * teSum) { same = true; break; }
                }
            }
            if (same)
            {
                r.Dispose();
                out_.Add(null);                    // looks the same as a turn already listed (a round symbol)
                continue;
            }
            var e = new Mat();
            r.ConvertTo(e, MatType.CV_32FC1);
            r.Dispose();
            var ink = new Mat();
            using (var ti = Rot90(tb, k)) ti.ConvertTo(ink, MatType.CV_32FC1);
            out_.Add(new Shape { E = e, Reach = Rot90(reach, k), Ink = ink, Own = (float)teSum, Solid = (float)solid, M = m });
        }
        return out_;
    }

    /// <summary>(cost, acceptable) of the symbol s at every position of the plan (or of box = (x0, y0, x1, y1) in it)</summary>
    private static (float[] Cost, bool[] Ok, int Rows, int Cols) CostOf(PlanIndex P, Shape s, (int X0, int Y0, int X1, int Y1)? box = null)
    {
        var (bx0, by0, bx1, by1) = box ?? (0, 0, P.Cols, P.Rows);
        int h = s.E.Rows, w = s.E.Cols;
        using var reachR = new Mat(P.Reach, new CvRect(bx0, by0, bx1 - bx0, by1 - by0));
        using var looseR = new Mat(P.Loose, new CvRect(bx0, by0, bx1 - bx0, by1 - by0));
        using var nearR = new Mat(P.Near, new CvRect(bx0, by0, bx1 - bx0, by1 - by0));
        using var fwd = new Mat(); using var bwd = new Mat(); using var cov = new Mat();
        // every stroke of the symbol close to a stroke of the plan ...
        Cv2.MatchTemplate(reachR, s.E, fwd, TemplateMatchModes.CCorr);
        int rows = fwd.Rows, cols = fwd.Cols;
        // ... the plan's strokes there close to the symbol's (a wire or wall through it costs nothing) ...
        var count = WindowSums(P.LooseSum, by0, bx0, h, w, rows, cols);
        Cv2.MatchTemplate(looseR, s.Reach, bwd, TemplateMatchModes.CCorr);
        var strokes = WindowSums(P.EdgesSum, by0, bx0, h, w, rows, cols);
        // ... and the symbol's own ink really there (a filled dot needs filled ink, not just a ring of strokes)
        Cv2.MatchTemplate(nearR, s.Ink, cov, TemplateMatchModes.CCorr);
        float* fp = F(fwd); float* bp = F(bwd); float* cp = F(cov);
        var cost = new float[rows * cols];
        var ok = new bool[rows * cols];
        float solid = Math.Max(s.Solid, 1f);
        float strokesMin = 0.6f * s.Own;
        for (int i = 0; i < rows * cols; i++)
        {
            float forward = fp[i] / s.Own;
            float backward = bp[i] / Math.Max(count[i], 1f);
            float cover = cp[i] / solid;
            cost[i] = forward + 0.5f * backward + 2f * (1f - Math.Min(cover, 1f));
            ok[i] = forward <= ForwardMax && backward <= BackwardMax && strokes[i] >= strokesMin && cover >= CoverMin;
        }
        return (cost, ok, rows, cols);
    }

    /// <summary>(kind, x, y, score) of every place a legend symbol is drawn (pixels), at any quarter turn; each place
    /// once. region: (x0, y0, x1, y1) to look in (the building); skip: pixel boxes to leave out (the legend).
    /// A quick search at a small size finds the places; each is then judged again in full detail against every legend
    /// symbol (at the small size, fine details such as a socket's three bars run together).</summary>
    internal static List<(string Kind, double X, double Y, double Score)> Match(Mat mask, Mat colour, List<Template> temps,
                                                                                (int X0, int Y0, int X1, int Y1) region, List<(int, int, int, int)> skip)
    {
        if (temps.Count == 0) return new();
        int H = mask.Rows, W = mask.Cols;
        using var full = Zeros(H, W, MatType.CV_8UC1);
        {
            byte* fp = B(full); byte* mp = B(mask);
            int rx0 = Math.Clamp(region.X0, 0, W), rx1 = Math.Clamp(region.X1, 0, W), ry0 = Math.Clamp(region.Y0, 0, H), ry1 = Math.Clamp(region.Y1, 0, H);
            for (int y = ry0; y < ry1; y++)
                for (int x = rx0; x < rx1; x++) fp[(long)y * W + x] = mp[(long)y * W + x];
            foreach (var (a, b, c, d) in skip)
                for (int y = Math.Max(0, b); y < Math.Min(H, d); y++)
                    for (int x = Math.Max(0, a); x < Math.Min(W, c); x++) fp[(long)y * W + x] = 0;
        }
        double typical = Py.Median(temps.Select(t => (double)t.Size));
        // only the part of the page with ink in it (plus a margin) is searched
        int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
        {
            byte* fp = B(full);
            for (int y = 0; y < H; y += 4)
                for (int x = 0; x < W; x += 4)
                    if (fp[(long)y * W + x] != 0)
                    {
                        int sy = y / 4, sx = x / 4;
                        minX = Math.Min(minX, sx); maxX = Math.Max(maxX, sx);
                        minY = Math.Min(minY, sy); maxY = Math.Max(maxY, sy);
                    }
        }
        if (maxX < 0) return new();
        int pad = (int)(2 * typical);
        int ox = Math.Max(0, minX * 4 - pad), oy = Math.Max(0, minY * 4 - pad);
        using var plan = Roi(full, ox, oy, Math.Min(W, maxX * 4 + 4 + pad), Math.Min(H, maxY * 4 + 4 + pad));
        // 1. the places: a quick search at a small size
        var cands = new List<(double Score, double X, double Y, double Size, int Ti, int K)>();
        using (var coarse = new PlanIndex(plan, Math.Clamp(SymbolPx / typical, 0.15, 1.0), SymbolPx))
        {
            for (int ti = 0; ti < temps.Count; ti++)
            {
                var shapes = Shapes(temps[ti], coarse.Fz);
                for (int k = 0; k < shapes.Count; k++)
                {
                    var s = shapes[k];
                    if (s == null) continue;
                    int eh = s.E.Rows, ew = s.E.Cols;
                    if (eh >= coarse.Rows || ew >= coarse.Cols) continue;
                    var (cost, ok, rows, cols) = CostOf(coarse, s);
                    using var costM = Mat.FromPixelData(rows, cols, MatType.CV_32FC1, cost);
                    using var win = Kernel(Math.Max(3, eh / 2) | 1, Math.Max(3, ew / 2) | 1);
                    using var er = new Mat();
                    Cv2.Erode(costM, er, win);
                    float* ep = F(er);
                    float lim = (float)(1.0 / CoarseMin - 1.0);
                    for (int yy = 0; yy < rows; yy++)
                        for (int xx = 0; xx < cols; xx++)
                        {
                            int i = yy * cols + xx;
                            if (cost[i] <= ep[i] + 1e-6f && ok[i] && cost[i] <= lim)
                                cands.Add((1.0 / (1.0 + cost[i]), (xx + ew / 2.0) / coarse.Fz, (yy + eh / 2.0) / coarse.Fz, (ew - 2 * s.M) / coarse.Fz, ti, k));
                        }
                }
                foreach (var s in shapes) s?.Dispose();
            }
        }
        var places = OneEach(cands.Select(c => (c.Score, c.X, c.Y, c.Size)).ToList(), typical).Take(MaxPlaces).ToList();
        // 2. each place judged again in detail - against the symbols (and turns) that came close there
        var found = new List<(double Score, double X, double Y, double Size, string Kind)>();
        using (var fine = new PlanIndex(plan, Math.Clamp(FinePx / typical, 0.15, 1.0), FinePx))
        {
            var allShapes = temps.Select(t => Shapes(t, fine.Fz)).ToList();
            var grid = new Dictionary<(long, long), List<int>>();
            for (int i = 0; i < cands.Count; i++)
            {
                var key = ((long)Py.FloorDiv(cands[i].X, typical), (long)Py.FloorDiv(cands[i].Y, typical));
                if (!grid.TryGetValue(key, out var l)) grid[key] = l = new List<int>();
                l.Add(i);
            }
            foreach (int pi in places)
            {
                var (_, cx, cy, size, _, _) = cands[pi];
                double fx = cx * fine.Fz, fy = cy * fine.Fz;
                var close = new List<int>();
                long gx = (long)Py.FloorDiv(cx, typical), gy = (long)Py.FloorDiv(cy, typical);
                foreach (var dx in new[] { -1, 0, 1 })
                    foreach (var dy in new[] { -1, 0, 1 })
                        if (grid.TryGetValue((gx + dx, gy + dy), out var l))
                            foreach (int j in l)
                                if (Math.Pow(cands[j].X - cx, 2) + Math.Pow(cands[j].Y - cy, 2) < Math.Pow(0.55 * Math.Max(size, cands[j].Size), 2))
                                    close.Add(j);
                close = Py.Sorted(close, j => -cands[j].Score);
                var tried = close.Select(j => (cands[j].Ti, cands[j].K)).Distinct().Take(4).ToList();   // the closest few
                (double C, int Ti, double X, double Y, double Size)? best = null;
                foreach (var (ti, k) in tried)
                {
                    if (k >= allShapes[ti].Count || allShapes[ti][k] == null) continue;
                    var s = allShapes[ti][k]!;
                    int h = s.E.Rows, w = s.E.Cols;
                    int slack = Math.Max(3, (int)(0.15 * Math.Max(h, w)));
                    int bx0 = (int)(fx - w / 2.0) - slack, by0 = (int)(fy - h / 2.0) - slack;
                    int bx1 = bx0 + w + 2 * slack, by1 = by0 + h + 2 * slack;
                    if (bx0 < 0 || by0 < 0 || bx1 > fine.Cols || by1 > fine.Rows) continue;
                    var (cost, ok, rows, cols) = CostOf(fine, s, (bx0, by0, bx1, by1));
                    int arg = 0;
                    float bc = float.PositiveInfinity;
                    for (int i = 0; i < cost.Length; i++)
                    {
                        float v = ok[i] ? cost[i] : float.PositiveInfinity;
                        if (v < bc) { bc = v; arg = i; }
                    }
                    int yy = arg / cols, xx = arg % cols;
                    double c = bc;
                    if (double.IsFinite(c) && (best == null || c < best.Value.C))
                        best = (c, ti, (bx0 + xx + w / 2.0) / fine.Fz, (by0 + yy + h / 2.0) / fine.Fz, (w - 2 * s.M) / fine.Fz);
                }
                if (best is not { } bb) continue;
                double score = 1.0 / (1.0 + bb.C);
                if (score < MatchMin) continue;
                var t = temps[bb.Ti];
                if (t.Colour != 0)                      // a coloured legend symbol: the match must not be another colour
                {
                    int hh = (int)(bb.Size / 2);
                    int gxp = (int)(bb.X + ox), gyp = (int)(bb.Y + oy);    // (in the whole picture)
                    var inks = new List<int>();
                    byte* cp = B(colour); byte* mp = B(mask);
                    for (int y = Math.Max(0, gyp - hh); y < Math.Min(H, gyp + hh); y++)
                        for (int x = Math.Max(0, gxp - hh); x < Math.Min(W, gxp + hh); x++)
                            if (mp[(long)y * W + x] > 0) inks.Add(cp[(long)y * W + x]);
                    // (its own colour among the coloured ink there, and not plain black / grey ink)
                    var coloured = inks.Where(v => v > 0).ToList();
                    if (coloured.Count == 0 || Mode(coloured) != t.Colour || coloured.Count < 0.3 * inks.Count) continue;
                }
                found.Add((score, bb.X, bb.Y, bb.Size, t.Kind));
            }
            foreach (var l in allShapes) foreach (var s in l) s?.Dispose();
        }
        var kept = OneEach(found.Select(f => (f.Score, f.X, f.Y, f.Size)).ToList(), typical);
        return kept.Select(i => (found[i].Kind, found[i].X + ox, found[i].Y + oy, found[i].Score)).ToList();
    }

    /// <summary>strongest first; a weaker place overlapping a kept one is the same symbol (or a part of it)</summary>
    private static List<int> OneEach(List<(double Score, double X, double Y, double Size)> places, double typical)
    {
        var order = Py.Sorted(Enumerable.Range(0, places.Count), i => -places[i].Score);
        var kept = new List<int>();
        var grid = new Dictionary<(long, long), List<int>>();
        double cell = Math.Max(4.0, typical);
        foreach (int i in order)
        {
            var (_, cx, cy, size) = places[i];
            long gx = (long)Py.FloorDiv(cx, cell), gy = (long)Py.FloorDiv(cy, cell);
            bool clash = false;
            foreach (var dx in new[] { -1, 0, 1 })
            {
                foreach (var dy in new[] { -1, 0, 1 })
                    if (grid.TryGetValue((gx + dx, gy + dy), out var l))
                        foreach (int j in l)
                            if (Math.Pow(places[j].X - cx, 2) + Math.Pow(places[j].Y - cy, 2) < Math.Pow(0.55 * Math.Max(size, places[j].Size), 2))
                            {
                                clash = true;
                                break;
                            }
                if (clash) break;
            }
            if (clash) continue;
            kept.Add(i);
            if (!grid.TryGetValue((gx, gy), out var g)) grid[(gx, gy)] = g = new List<int>();
            g.Add(i);
        }
        return kept;
    }

    /// <summary>a room size misread by OCR ("I89 X 14-5" for 18'-9" x 14'-6"): a side over 60 ft</summary>
    private static bool AbsurdSize(string text)
    {
        var d = Common.ParseDimensions(text);
        if (d == null) return false;
        var nums = Regex.Matches(text, @"\d+(?:\.\d+)?", Py.RX).Select(m => Py.ParseFloat(m.Value)).ToList();
        var check = nums.Take(1).Concat(nums.Count >= 2 ? new[] { nums[^2] } : Array.Empty<double>());
        return d.Value.Sqft > 3600 || check.Any(v => v > 60);
    }

    private static readonly Regex SingleFeet = new(@"^(\d{2,3})\s*(?:'|’|′|FT)\s*(?:-?\s*(\d{1,2})\s*(?:""|''|”|″)?)?$", Py.RX);
    private const double TypicalRoomSqft = 150.0;

    /// <summary>(1:N, how it was worked out) for a picture whose print size is unknown, or (null, "") to leave it to the
    /// room dimensions written on the plan. An overall dimension written on the plan ("50'", "90'") is taken as the
    /// building's length; else a typical room (150 sq ft) sets the size.</summary>
    internal static (int? Scale, string How) PictureScale(List<(TextItem T, PRect R)> linesPt, (int X0, int Y0, int X1, int Y1)? footprint,
                                                          List<double> roomsPx, double pt, List<PRect>? viewsPt = null)
    {
        var values = new List<(double Feet, bool Upright, double CX, double CY)>();
        foreach (var (t, r) in linesPt)
        {
            var m = SingleFeet.Match(t.Text.Trim().ToUpperInvariant());
            if (m.Success)
                values.Add((Py.ParseInt(m.Groups[1].Value) + (m.Groups[2].Success ? Py.ParseInt(m.Groups[2].Value) : 0) / 12.0,
                            r.Height > r.Width, (r.X0 + r.X1) / 2, (r.Y0 + r.Y1) / 2));
        }
        values = values.Where(v => 15 <= v.Feet && v.Feet <= 400).ToList();
        if (values.Count > 0 && viewsPt is { Count: > 0 })
        {
            // several drawings on the sheet: the dimension spans the drawing it is written in (drawn to the same scale)
            var desc = values.Select((v, i) => (v, i)).OrderByDescending(x => x.v.Feet).ThenByDescending(x => x.v.Upright)
                             .ThenByDescending(x => x.v.CX).ThenByDescending(x => x.v.CY).ThenBy(x => x.i).Select(x => x.v);
            foreach (var (feet, upright, cx, cy) in desc)
            {
                // (inside it, or written just outside its edge - a dimension line above or beside the drawing)
                var vv = viewsPt.Where(b => b.X0 - 0.08 * (b.X1 - b.X0) <= cx && cx <= b.X1 + 0.08 * (b.X1 - b.X0)
                                            && b.Y0 - 0.08 * (b.Y1 - b.Y0) <= cy && cy <= b.Y1 + 0.08 * (b.Y1 - b.Y0)).Select(b => (PRect?)b).FirstOrDefault();
                if (vv is { } v)
                {
                    double side = 0.85 * (upright ? v.Y1 - v.Y0 : v.X1 - v.X0);    // (less its margin and texts)
                    int n = Py.RoundInt(feet * 864 / Math.Max(side, 1));
                    if (20 <= n && n <= 2000) return (n, $"from the dimension {Py.G(feet)}' written on the sheet");
                }
            }
        }
        if (values.Count > 0 && footprint is { } fp && !(viewsPt is { Count: > 0 }))
        {
            double pad = 0.02 * Math.Max(fp.X1 - fp.X0, fp.Y1 - fp.Y0) / 0.96;
            var top = values.OrderByDescending(x => x.Feet).ThenByDescending(x => x.Upright).ThenByDescending(x => x.CX).ThenByDescending(x => x.CY).First();
            double sidePt = (top.Upright ? fp.Y1 - fp.Y0 : fp.X1 - fp.X0) - 2 * pad;
            int n = Py.RoundInt(top.Feet * 864 / Math.Max(sidePt * pt, 1));
            if (20 <= n && n <= 2000) return (n, $"from the overall dimension {Py.G(top.Feet)}' written on the plan");
        }
        int dims = linesPt.Count(l => Common.ParseDimensions(l.T.Text) != null);
        if (dims >= 2 || roomsPx.Count == 0) return (null, "");
        double ftPt = Math.Pow(Py.Median(roomsPx) * pt * pt / TypicalRoomSqft, 0.5);
        int n2 = Py.RoundInt(864 / Math.Max(ftPt, 1e-6));
        return 20 <= n2 && n2 <= 2000 ? (n2, "estimated from the number and size of the rooms (no sizes could be read)") : (null, "");
    }

    /// <summary>UnnamedRoomsFrom on the walls; when they enclose (almost) nothing - walls drawn as thin double lines -
    /// from all the drawing's lines</summary>
    internal static List<(TextItem T, PRect R)> UnnamedRooms(Mat walls, (int X0, int Y0, int X1, int Y1)? footprint, List<(TextItem T, PRect R)> linesPt,
                                                             double pt, Mat? mask = null, double linePx = 0.0, List<double>? roomsPx = null,
                                                             List<List<Pt>>? outlines = null)
    {
        roomsPx ??= new();
        outlines ??= new();
        var found = UnnamedRoomsFrom(walls, footprint, linesPt, pt, mask, linePx, roomsPx, outlines);
        if (roomsPx.Count < 2 && mask != null && footprint != null)
        {
            roomsPx.Clear();
            outlines.Clear();
            found = UnnamedRoomsFrom(mask, footprint, linesPt, pt, null, 0.0, roomsPx, outlines);
        }
        return found;
    }

    /// <summary>Spaces enclosed by solid walls that have no room name read in them (the names too small or blurred to
    /// read): a text line "ROOM n" at a point inside each, so they are traced and counted as rooms all the same.
    /// roomsPx gets the area in pixels of every room space found, named or not.</summary>
    private static List<(TextItem T, PRect R)> UnnamedRoomsFrom(Mat walls, (int X0, int Y0, int X1, int Y1)? footprint, List<(TextItem T, PRect R)> linesPt,
                                                                double pt, Mat? mask, double linePx, List<double> roomsPx, List<List<Pt>> outlines)
    {
        if (footprint is not { } fp) return new();
        var (x0, y0, x1, y1) = fp;
        using var wroi = Roi(walls, x0, y0, x1, y1);
        int h = wroi.Rows, ww = wroi.Cols;
        using var w = new Mat(h, ww, MatType.CV_8UC1);
        byte* src = B(wroi); byte* wp = B(w);
        for (int i = 0; i < h * ww; i++) wp[i] = src[i] > 0 ? (byte)1 : (byte)0;
        if (linePx != 0 && mask != null)
        {
            // a blurry, enlarged picture makes every line look thick, so the solid-wall test can miss the walls: here
            // walls are ink clearly heavier than the usual line that runs long (furniture and text are short)
            using var mroi = Roi(mask, x0, y0, x1, y1);
            int ks = Math.Max(3, (int)(1.6 * linePx));
            using var ek = Ellipse(ks, ks);
            using var heavy = Morph(mroi, MorphTypes.Open, ek);
            using var cc = CC(heavy);
            var keep = new bool[cc.N];
            for (int i = 1; i < cc.N; i++) keep[i] = Math.Max(cc.W(i), cc.H(i)) >= 0.06 * Math.Min(heavy.Rows, heavy.Cols);
            int* lp = I(cc.Labels);
            for (int i = 0; i < h * ww; i++) if (keep[lp[i]]) wp[i] = 1;
        }
        long wsum = 0;
        for (int i = 0; i < h * ww; i++) wsum += wp[i];
        if (wsum < 100) return new();
        // door openings closed: about four wall thicknesses (a 9" wall, a 3' door)
        using var dist = new Mat();
        Cv2.DistanceTransform(w, dist, DistanceTypes.L2, DistanceTransformMasks.Mask3);
        using var k3 = Kernel(3, 3);
        using var distD = Dilate(dist, k3);
        float* dp = F(dist); float* ddp = F(distD);
        var ridge = new List<float>();
        for (int i = 0; i < h * ww; i++) if (wp[i] > 0 && dp[i] >= ddp[i] - 1e-6f) ridge.Add(dp[i]);
        double thick = ridge.Count > 0 ? 2 * (double)Py.MedianF(ridge) : 2.0;     // (no ridge measured: the usual line)
        if (linePx != 0) thick = Math.Min(thick, 2 * linePx);      // (furniture joined to a wall makes it look thicker)
        int k = Math.Max(5, (int)(4.5 * Math.Max(thick, 2))) | 1;
        using var rk = RectK(k, k);
        using var closed = Morph(w, MorphTypes.Close, rk);
        using var free = new Mat(h, ww, MatType.CV_8UC1);
        byte* clp = B(closed); byte* fp2 = B(free);
        for (int i = 0; i < h * ww; i++) fp2[i] = clp[i] == 0 ? (byte)1 : (byte)0;
        using var fcc = CC(free, PixelConnectivity.Connectivity4);
        long total = (long)h * ww;
        var named = linesPt.Where(l => Common.RoomType(l.T.Text) != null)
                           .Select(l => (((l.R.X0 + l.R.X1) / 2) / pt - x0, ((l.R.Y0 + l.R.Y1) / 2) / pt - y0)).ToList();
        var out_ = new List<(double, double)>();
        roomsPx.Clear();
        int* flab = I(fcc.Labels);
        for (int i = 1; i < fcc.N; i++)
        {
            int x = fcc.X(i), y = fcc.Y(i), bw = fcc.W(i), bh = fcc.H(i), area = fcc.Area(i);
            if (x == 0 || y == 0 || x + bw >= ww || y + bh >= h) continue;        // open to the outside of the building
            if (!(0.004 * total <= area && area <= 0.4 * total) || Math.Min(bw, bh) < 0.8 * k) continue;   // a duct or a gap, or not a room
            roomsPx.Add(area);
            using var cell = new Mat(bh, bw, MatType.CV_8UC1, Scalar.All(0));
            byte* cp = B(cell);
            for (int yy = 0; yy < bh; yy++)
                for (int xx = 0; xx < bw; xx++)
                    if (flab[(long)(y + yy) * ww + x + xx] == i) cp[yy * bw + xx] = 1;
            Cv2.FindContours(cell, out CvPoint[][] cs, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            if (cs.Length > 0)
            {
                var biggest = Py.MaxBy(cs, c => Cv2.ContourArea(c));
                var c = Cv2.ApproxPolyDP(biggest, 2.0, true);
                if (c.Length >= 3) outlines.Add(c.Select(q => new Pt((x0 + x + q.X) * pt, (y0 + y + q.Y) * pt)).ToList());
            }
            bool hasName = named.Any(nm =>
            {
                int py = (int)nm.Item2 - y, px = (int)nm.Item1 - x;
                return 0 <= py && py < bh && 0 <= px && px < bw && cp[py * bw + px] != 0;
            });
            if (hasName) continue;                     // its name was read
            using var padded = new Mat();
            Cv2.CopyMakeBorder(cell, padded, 1, 1, 1, 1, BorderTypes.Constant, Scalar.All(0));
            using var dfull = new Mat();
            Cv2.DistanceTransform(padded, dfull, DistanceTypes.L2, DistanceTransformMasks.Mask3);
            using var d = Roi(dfull, 1, 1, bw + 1, bh + 1);
            float* dq = F(d);
            int arg = 0;
            float bv = float.NegativeInfinity;
            for (int q = 0; q < bh * bw; q++) if (dq[q] > bv) { bv = dq[q]; arg = q; }    // the point deepest inside it
            int cy = arg / bw, cx = arg % bw;
            out_.Add(((x0 + x + cx) * pt, (y0 + y + cy) * pt));
        }
        var lines = new List<(TextItem, PRect)>();
        int j = 0;
        foreach (var (px, py) in out_.OrderBy(p => (long)Py.Round(p.Item2 / 50)).ThenBy(p => p.Item1))
        {
            j++;
            double hh = 4.0;
            lines.Add((new TextItem($"ROOM {j}", px, py, hh), new PRect(px - 2 * hh, py - hh / 2, px + 2 * hh, py + hh / 2)));
        }
        return lines;
    }

    /// <summary>for each polyline, a label shared by all polylines connected to it end to end (lines meeting at a
    /// junction end at exactly the same point)</summary>
    private static int[] JoinedLabels(List<Pt[]> polylines)
    {
        var parent = Enumerable.Range(0, polylines.Count).ToArray();
        int Root(int i)
        {
            while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
            return i;
        }
        var at = new Dictionary<(long, long), int>();
        for (int i = 0; i < polylines.Count; i++)
        {
            var p = polylines[i];
            foreach (var q in new[] { p[0], p[^1] })
            {
                var key = ((long)Math.Round(q.X * 2, MidpointRounding.ToEven), (long)Math.Round(q.Y * 2, MidpointRounding.ToEven));
                if (!at.TryGetValue(key, out int j)) { at[key] = i; j = i; }
                if (j != i) parent[Root(i)] = Root(j);
            }
        }
        return Enumerable.Range(0, polylines.Count).Select(Root).ToArray();
    }

    internal sealed record TracedPage(int Width, int Height, double Pt, List<(List<Pt[]> Pieces, (double, double, double) Col)> Filled,
                                      List<(Pt[] Pts, double Width, (double, double, double) Col)> Lines);

    /// <summary>the traced drawings, one page each</summary>
    internal static byte[] ToPdf(IList<TracedPage> pages)
    {
        // the page's drawing commands are written directly (PDF y runs upwards)
        using var doc = new PdfDoc();
        foreach (var tp in pages)
        {
            double pt = tp.Pt;
            double height = tp.Height * pt;
            var page = doc.NewPage(tp.Width * pt, height);
            var sb = new StringBuilder();
            void Path(Pt[] p)
            {
                for (int i = 0; i < p.Length; i++)
                {
                    if (i > 0) sb.Append(' ');
                    sb.Append(Py.FixedFast(p[i].X * pt, 2)).Append(' ').Append(Py.FixedFast(height - p[i].Y * pt, 2)).Append(i == 0 ? " m" : " l");
                }
            }
            var ops = new List<string> { "q 1 J 1 j" };
            foreach (var (pieces, col) in tp.Filled)            // one path per piece of thick ink (a wall network, a dot)
            {
                sb.Clear();
                sb.Append($"{Py.FixedFast(col.Item1, 2)} {Py.FixedFast(col.Item2, 2)} {Py.FixedFast(col.Item3, 2)} rg ");
                for (int i = 0; i < pieces.Count; i++)
                {
                    if (i > 0) sb.Append(' ');
                    Path(pieces[i]);
                    sb.Append(" h");
                }
                sb.Append(" f");
                ops.Add(sb.ToString());
            }
            // lines that meet (and share weight and colour) are one path, like a wall drawn in one go in CAD: rooms
            // are traced from long paths only, and the trace cuts a wall wherever something touches it
            var roots = JoinedLabels(tp.Lines.Select(l => l.Pts).ToList());
            var groups = new Dictionary<(int, double, (double, double, double)), List<Pt[]>>();
            var order = new List<(int, double, (double, double, double))>();
            for (int i = 0; i < tp.Lines.Count; i++)
            {
                var key = (roots[i], tp.Lines[i].Width, tp.Lines[i].Col);
                if (!groups.TryGetValue(key, out var l)) { groups[key] = l = new List<Pt[]>(); order.Add(key); }
                l.Add(tp.Lines[i].Pts);
            }
            foreach (var key in order)
            {
                var (_, width, col) = key;
                sb.Clear();
                sb.Append($"{Py.FixedFast(col.Item1, 2)} {Py.FixedFast(col.Item2, 2)} {Py.FixedFast(col.Item3, 2)} RG {Py.FixedFast(width * pt, 3)} w ");
                var parts = groups[key];
                for (int i = 0; i < parts.Count; i++)
                {
                    if (i > 0) sb.Append(' ');
                    Path(parts[i]);
                }
                sb.Append(" S");
                ops.Add(sb.ToString());
            }
            ops.Add("Q");
            doc.SetContents(page, Encoding.ASCII.GetBytes(string.Join("\n", ops)));
        }
        return doc.ToBytes(0, true);
    }

    /// <summary>kind: "scan" (a PDF of scanned pages) or "image". Returns the PDF reader's result plus the traced PDF.</summary>
    public static (Dictionary<string, string> Project, List<Sheet> Sheets, List<LegendRow> Legend, byte[] Traced) Read(
        byte[] data, string kind, SymbolMapper mapper, List<string> log, List<string> warnings, IList<int>? pages = null)
    {
        var pics = Pictures(data, kind, pages);
        var what = kind == "scan" ? "Scanned PDF" : "Picture";
        log.Add($"{what}, {pics.Count} page(s) - read from the image: words by OCR, lines and walls traced from the ink");
        var traced = new List<TracedPage>();
        var ocrLines = new Dictionary<int, List<(TextItem T, PRect R)>>();
        var pointsOf = new Dictionary<int, List<PointItem>>();
        var outlinesOf = new Dictionary<int, List<List<Pt>>>();
        bool unknownSize = false;
        var smallPages = new List<int>(); var unreadable = new List<int>(); var unnamedPages = new List<int>(); var scaleNotes = new List<string>();
        try
        {
            for (int pi = 0; pi < pics.Count; pi++)
            {
                var (pageNo, img, pt, known, enlarged) = pics[pi];
                Ocr.Progress?.Invoke(pi + 1, pics.Count);
                var words = ReadWords(img, enlarged);
                if (enlarged) smallPages.Add(pageNo);
                if (words.Count(w => w.Text.Trim().Length >= 4) < 5) unreadable.Add(pageNo);
                var (filled, lines, st) = Vectorise(img, words);
                using (st)
                {
                    var linesPt = Ocr.ToLines(words, pt).Where(l => !AbsurdSize(l.T.Text)).ToList();
                    if (!known)      // the paper size is unknown, so a written "SCALE 1:100" can't give sizes
                    {
                        linesPt = linesPt.Where(l => !Common.Norm(l.T.Text).Contains("SCALE")).ToList();
                        unknownSize = true;
                    }
                    int n = traced.Count + 1;
                    ocrLines[n] = linesPt;
                    var tracedPage = new TracedPage(img.Cols, img.Rows, pt, filled, lines);
                    traced.Add(tracedPage);
                    // symbols: the legend rows' symbols, matched as pictures over the building
                    var rows = PdfReader.LegendRows(linesPt, mapper);
                    var temps = Templates(st.Mask, st.Colour, rows, pt);
                    var found = new List<(string Kind, double X, double Y, double Score)>();
                    try
                    {
                        if (temps.Count > 0)
                        {
                            int H = st.Mask.Rows, W = st.Mask.Cols;
                            var region = st.Footprint ?? (0, 0, W, H);
                            var legendPx = new List<(int, int, int, int)>();
                            foreach (var (_, _, strip) in rows)       // the legend table itself (and a margin) is not the plan
                                legendPx.Add(((int)(strip.X0 / pt) - 10, (int)(strip.Y0 / pt) - 10, (int)(strip.X1 / pt) + 10, (int)(strip.Y1 / pt) + 10));
                            if (legendPx.Count > 0)
                            {
                                int xs0 = legendPx.Min(b => b.Item1), ys0 = legendPx.Min(b => b.Item2);
                                int xs1 = legendPx.Max(b => b.Item3), ys1 = legendPx.Max(b => b.Item4);
                                legendPx = new() { (xs0, ys0, xs1 + (int)(0.4 * (xs1 - xs0)), ys1) };    // (with the description column)
                            }
                            found = Match(st.SymbolInk, st.Colour, temps, region, legendPx);
                            pointsOf[n] = found.Select(f => new PointItem(f.Kind, f.X * pt, f.Y * pt)).ToList();
                            // short texts inside a matched symbol ("TV" in its box, "AC") are part of it: not room names,
                            // and not counted again as text symbols
                            double reach = Math.Pow(0.6 * Py.Median(temps.Select(t => (double)t.Size)), 2);
                            ocrLines[n] = linesPt.Where(l => l.T.Text.Trim().Length > 4 || !found.Any(f =>
                                Math.Pow((l.R.X0 + l.R.X1) / 2 / pt - f.X, 2) + Math.Pow((l.R.Y0 + l.R.Y1) / 2 / pt - f.Y, 2) < reach)).ToList();
                        }
                    }
                    finally
                    {
                        foreach (var t in temps) t.Dispose();
                    }
                    var roomsPx = new List<double>();
                    outlinesOf[n] = new();
                    // several drawings on the sheet (floor plans side by side, elevations, an area plan): rooms are looked
                    // for in each floor plan on its own - not across the sheet, and not in elevations or sections
                    var regions = new List<(int, int, int, int)?> { st.Footprint };
                    List<Stroke> pageItems;
                    using (var single = new PdfDoc(ToPdf(new[] { tracedPage })))
                        pageItems = PdfGeometry.Strokes(single[0].GetDrawings());
                    var views = PdfReader.Views(pageItems, ocrLines[n], new PRect(0, 0, img.Cols * pt, img.Rows * pt), null);
                    if (views.Count >= 2 && views.Count(v => v.Titled) >= 2)
                        regions = views.Where(v => !PdfReader.NotPlan.Contains(v.Kind) && v.Kind != "area")
                                       .Select(v => ((int, int, int, int)?)((int)(v.Box.X0 / pt), (int)(v.Box.Y0 / pt), (int)(v.Box.X1 / pt), (int)(v.Box.Y1 / pt))).ToList();
                    var extra = new List<(TextItem T, PRect R)>();
                    foreach (var region in regions)
                    {
                        var sizes = new List<double>();
                        var shapesHere = new List<List<Pt>>();
                        extra.AddRange(UnnamedRooms(st.Walls, region, ocrLines[n], pt, st.Mask, st.LinePx, sizes, shapesHere));
                        roomsPx.AddRange(sizes);
                        outlinesOf[n].AddRange(shapesHere);
                    }
                    if (regions.Count > 1)       // numbered across the sheet (each plan is its own floor anyway)
                        extra = extra.Select((x, j) => (new TextItem($"ROOM {j + 1}", x.T.X, x.T.Y, x.T.H), x.R)).ToList();
                    if (extra.Count > 0)
                    {
                        ocrLines[n] = ocrLines[n].Concat(extra).ToList();
                        unnamedPages.Add(pageNo);
                    }
                    // the drawing's scale, when the picture can't give it (no print size, or no "SCALE 1:100" read)
                    if (!known || !ocrLines[n].Any(l => Common.Norm(l.T.Text).Contains("SCALE") && l.T.Text.Contains(':')))
                    {
                        var (scale, how) = PictureScale(ocrLines[n], st.Footprint, roomsPx, pt, regions.Count > 1 ? views.Select(v => v.Box).ToList() : null);
                        if (scale is { } sc && sc != 0)
                        {
                            ocrLines[n] = ocrLines[n].Where(l => !Common.Norm(l.T.Text).Contains("SCALE")).ToList();
                            ocrLines[n].Add((new TextItem($"SCALE 1:{sc}", 1, 1, 1), new PRect(0, 0, 2, 2)));
                            scaleNotes.Add($"Page {pageNo}: room sizes {how} - check them.");
                            unknownSize = false;
                        }
                    }
                    var counts = found.GroupBy(f => f.Kind).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Key} {g.Count()}").ToList();
                    log.Add($"Page {pageNo}: {words.Count} word(s) read, {lines.Count} line(s) and {filled.Count} solid shape(s) " +
                            $"traced (lines about {Py.Fixed(st.LinePx, 1)} px wide); {temps.Count} legend symbol(s) cut out and matched: " +
                            (counts.Count > 0 ? string.Join(", ", counts) : "none found"));
                }
            }
        }
        finally
        {
            foreach (var p in pics) p.Img.Dispose();
        }
        var pdf = ToPdf(traced);
        var (project, sheets, legend) = PdfReader.Read(pdf, mapper, log, warnings, null, ocrLines, pointsOf, outlinesOf);
        if (scaleNotes.Count > 0) project.Remove("scale");          // (our own estimate, not written on the drawing)
        warnings.Insert(0, $"This drawing was read from a {(kind == "scan" ? "scan" : "picture")}: words by OCR and " +
                           "lines traced from the image - check room names and counts (a DXF or CAD PDF reads best).");
        if (unknownSize)
            warnings.Insert(1, "The picture doesn't say its print size, so room areas come from the dimensions written " +
                               "in the rooms (or an estimate) - check the areas.");
        if (unreadable.Count > 0)
            warnings.Insert(0, $"Page(s) {PdfReader.PagesText(unreadable)}: almost no words could be read - the picture is too small " +
                               "or blurry. Send the original file (DXF / PDF from CAD) or a sharp scan of at least " +
                               "2000 pixels across.");
        for (int i = scaleNotes.Count - 1; i >= 0; i--) warnings.Insert(0, scaleNotes[i]);
        if (unnamedPages.Count > 0)
            warnings.Insert(0, $"Page(s) {PdfReader.PagesText(unnamedPages)}: some rooms' names could not be read - they are listed " +
                               "as ROOM 1, ROOM 2 ...; rename them and check their sizes.");
        if (unreadable.Count == 0 && smallPages.Count > 0)
            warnings.Insert(0, $"Page(s) {PdfReader.PagesText(smallPages)}: the picture is small, so small text may be misread. " +
                               "WhatsApp shrinks photos - send the drawing as a Document, or the original file.");
        return (project, sheets, legend, pdf);
    }
}
