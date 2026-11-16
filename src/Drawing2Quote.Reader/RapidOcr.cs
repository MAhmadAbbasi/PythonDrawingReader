using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Drawing2Quote.Reader;

/// <summary>
/// RapidOCR (rapidocr-onnxruntime 1.4) on ONNX Runtime: the PP-OCRv4 text detector and recogniser with the same
/// pre- and post-processing, so words are found and read as the Python reader finds and reads them.
/// Images are RGB, given to the models as they are (as the Python reader gives its numpy arrays).
/// </summary>
internal sealed class RapidOcr : IDisposable
{
    private readonly InferenceSession _det, _rec;
    private readonly string _detInput, _recInput;
    private readonly string[] _chars;
    private readonly object _lock = new();

    // config.yaml
    private const int MaxSideLen = 2000, MinSideLen = 30, MinHeight = 30;
    private const double WidthHeightRatio = 8;
    private const int LimitSideLen = 736;
    private const float Thresh = 0.3f;
    private const double BoxThresh = 0.5, UnclipRatio = 1.6;
    private const int MaxCandidates = 1000, MinSize = 3;
    private const int RecH = 48, RecW = 320, RecBatch = 6;

    public RapidOcr(string modelDir)
    {
        var opts = new SessionOptions
        {
            EnableCpuMemArena = false,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _det = new InferenceSession(Path.Combine(modelDir, "ch_PP-OCRv4_det_infer.onnx"), opts);
        _rec = new InferenceSession(Path.Combine(modelDir, "ch_PP-OCRv4_rec_infer.onnx"), opts);
        _detInput = _det.InputMetadata.Keys.First();
        _recInput = _rec.InputMetadata.Keys.First();
        // the character list is kept in the model; blank first and a space last (CTCLabelDecode)
        var meta = _rec.ModelMetadata.CustomMetadataMap["character"];
        var chars = Py.SplitLines(meta);
        chars.Add(" ");
        chars.Insert(0, "blank");
        _chars = chars.ToArray();
    }

    public void Dispose()
    {
        _det.Dispose();
        _rec.Dispose();
    }

    // ── detection: RapidOCR()(img, use_det=True, use_cls=False, use_rec=False) ──
    /// <summary>the text boxes found in an RGB image: 4 corners each (tl, tr, br, bl), in the image's pixels</summary>
    public List<float[,]> Detect(Mat img)
    {
        int rawH = img.Rows, rawW = img.Cols;
        // preprocess: max / min side
        var work = img;
        double ratioH = 1.0, ratioW = 1.0;
        if (Math.Max(work.Rows, work.Cols) > MaxSideLen) (work, ratioH, ratioW) = ResizeSide(work, MaxSideLen, reduce: true);
        if (Math.Min(work.Rows, work.Cols) < MinSideLen) (work, ratioH, ratioW) = ResizeSide(work, MinSideLen, reduce: false);
        // letterbox for very flat images
        int padTop = 0;
        {
            int h = work.Rows, w = work.Cols;
            bool limit = (double)w / h > WidthHeightRatio;
            if (h <= MinHeight || limit)
            {
                int newH = Math.Max((int)(w / WidthHeightRatio), MinHeight) * 2;
                int pad = (int)(Math.Abs(newH - h) / 2.0);
                var padded = new Mat();
                Cv2.CopyMakeBorder(work, padded, pad, pad, 0, 0, BorderTypes.Constant, Scalar.All(0));
                work = padded;
                padTop = pad;
            }
        }
        var boxes = TextDet(work);
        if (boxes.Count == 0) return boxes;
        boxes = SortedBoxes(boxes);
        // back to the image's own pixels
        foreach (var b in boxes)
        {
            for (int p = 0; p < 4; p++)
            {
                float x = b[p, 0] - 0f, y = b[p, 1] - padTop;
                x = (float)(x * (float)ratioW);
                y = (float)(y * (float)ratioH);
                if (x < 0) x = 0;
                if (y < 0) y = 0;
                if (x > rawW) x = rawW;
                if (y > rawH) y = rawH;
                b[p, 0] = x; b[p, 1] = y;
            }
        }
        return boxes;
    }

    private static (Mat, double, double) ResizeSide(Mat img, int side, bool reduce)
    {
        int h = img.Rows, w = img.Cols;
        double ratio = 1.0;
        if (reduce ? Math.Max(h, w) > side : Math.Min(h, w) < side)
            ratio = reduce ? (h > w ? (double)side / h : (double)side / w) : (h < w ? (double)side / h : (double)side / w);
        int rh = (int)(h * ratio), rw = (int)(w * ratio);
        rh = (int)(Py.Round(rh / 32.0) * 32);
        rw = (int)(Py.Round(rw / 32.0) * 32);
        if (rw <= 0 || rh <= 0) throw new InvalidOperationException("resize_w or resize_h is less than or equal to 0");
        var dst = new Mat();
        Cv2.Resize(img, dst, new Size(rw, rh));
        return (dst, (double)h / rh, (double)w / rw);
    }

    private List<float[,]> TextDet(Mat img)
    {
        int oh = img.Rows, ow = img.Cols;
        // DetPreProcess: the short side to 736, both sides multiples of 32
        double ratio = 1.0;
        if (Math.Min(oh, ow) < LimitSideLen) ratio = oh < ow ? (double)LimitSideLen / oh : (double)LimitSideLen / ow;
        int rh = (int)(oh * ratio), rw = (int)(ow * ratio);
        rh = (int)(Py.Round(rh / 32.0) * 32);
        rw = (int)(Py.Round(rw / 32.0) * 32);
        if (rw <= 0 || rh <= 0) return new List<float[,]>();
        using var resized = new Mat();
        Cv2.Resize(img, resized, new Size(rw, rh));
        var input = new DenseTensor<float>(new[] { 1, 3, rh, rw });
        float scale = (float)(1 / 255.0);
        var idx = resized.GetGenericIndexer<Vec3b>();
        for (int y = 0; y < rh; y++)
            for (int x = 0; x < rw; x++)
            {
                var v = idx[y, x];
                for (int c = 0; c < 3; c++)
                {
                    float f = v[c] * scale;
                    input[0, c, y, x] = (float)(((double)f - 0.5) / 0.5);
                }
            }
        float[] pred;
        int ph, pw;
        lock (_lock)
        {
            using var results = _det.Run(new[] { NamedOnnxValue.CreateFromTensor(_detInput, input) });
            var t = results.First().AsTensor<float>();
            ph = t.Dimensions[2]; pw = t.Dimensions[3];
            pred = t.ToArray();
        }
        // DBPostProcess
        using var predMat = Mat.FromPixelData(ph, pw, MatType.CV_32FC1, pred);
        using var seg = new Mat(ph, pw, MatType.CV_8UC1);
        var si = seg.GetGenericIndexer<byte>();
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++)
                si[y, x] = pred[y * pw + x] > Thresh ? (byte)1 : (byte)0;
        using var mask = new Mat();
        using (var kernel = Mat.Ones(2, 2, MatType.CV_8UC1).ToMat())
            Cv2.Dilate(seg, mask, kernel);
        var boxes = BoxesFromBitmap(predMat, pred, pw, mask, ow, oh);
        // filter_tag_det_res
        var out_ = new List<float[,]>();
        foreach (var box in boxes)
        {
            var b = OrderPointsClockwise(box);
            for (int p = 0; p < 4; p++)
            {
                b[p, 0] = (int)Math.Min(Math.Max(b[p, 0], 0), ow - 1);
                b[p, 1] = (int)Math.Min(Math.Max(b[p, 1], 0), oh - 1);
            }
            int rectW = (int)NormF(b[0, 0] - b[1, 0], b[0, 1] - b[1, 1]);
            int rectH = (int)NormF(b[0, 0] - b[3, 0], b[0, 1] - b[3, 1]);
            if (rectW <= 3 || rectH <= 3) continue;
            out_.Add(b);
        }
        return out_;
    }

    private static float NormF(float dx, float dy) => MathF.Sqrt(dx * dx + dy * dy);

    private List<int[,]> BoxesFromBitmap(Mat predMat, float[] pred, int pw, Mat bitmap, int destW, int destH)
    {
        int height = bitmap.Rows, width = bitmap.Cols;
        using var bin = new Mat();
        bitmap.ConvertTo(bin, MatType.CV_8UC1, 255);
        Cv2.FindContours(bin, out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        int n = Math.Min(contours.Length, MaxCandidates);
        var boxes = new List<int[,]>();
        for (int index = 0; index < n; index++)
        {
            var contour = contours[index];
            var (points, sside) = GetMiniBoxes(contour);
            if (sside < MinSize) continue;
            double score = BoxScoreFast(predMat, height, width, points);
            if (BoxThresh > score) continue;
            var expanded = Unclip(points);
            if (expanded.Count == 0) continue;
            var (box, sside2) = GetMiniBoxes(expanded.Select(p => new Point((int)p.X, (int)p.Y)).ToArray());
            if (sside2 < MinSize + 2) continue;
            var ib = new int[4, 2];
            for (int p = 0; p < 4; p++)
            {
                float bx = (float)Math.Round((double)(float)(box[p].X / (float)width * destW), MidpointRounding.ToEven);
                float by = (float)Math.Round((double)(float)(box[p].Y / (float)height * destH), MidpointRounding.ToEven);
                ib[p, 0] = (int)Math.Clamp(bx, 0, destW);
                ib[p, 1] = (int)Math.Clamp(by, 0, destH);
            }
            boxes.Add(ib);
        }
        return boxes;
    }

    private static (Point2f[] Box, float SSide) GetMiniBoxes(Point[] contour)
    {
        var rect = Cv2.MinAreaRect(contour);
        var pts = Py.Sorted(rect.Points(), p => p.X);
        int i1, i2, i3, i4;
        if (pts[1].Y > pts[0].Y) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
        if (pts[3].Y > pts[2].Y) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
        return (new[] { pts[i1], pts[i2], pts[i3], pts[i4] }, Math.Min(rect.Size.Width, rect.Size.Height));
    }

    private static double BoxScoreFast(Mat bitmap, int h, int w, Point2f[] box)
    {
        int xmin = Math.Clamp((int)MathF.Floor(box.Min(p => p.X)), 0, w - 1);
        int xmax = Math.Clamp((int)MathF.Ceiling(box.Max(p => p.X)), 0, w - 1);
        int ymin = Math.Clamp((int)MathF.Floor(box.Min(p => p.Y)), 0, h - 1);
        int ymax = Math.Clamp((int)MathF.Ceiling(box.Max(p => p.Y)), 0, h - 1);
        using var mask = new Mat(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1, Scalar.All(0));
        var poly = box.Select(p => new Point((int)(p.X - xmin), (int)(p.Y - ymin))).ToArray();
        Cv2.FillPoly(mask, new[] { poly }, Scalar.All(1));
        using var roi = new Mat(bitmap, new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
        return Cv2.Mean(roi, mask).Val0;
    }

    /// <summary>the box grown by area * 1.6 / perimeter: pyclipper (Clipper 6.4.2) offset, round joins</summary>
    private static List<(long X, long Y)> Unclip(Point2f[] box)
    {
        // shapely: area and length of the box polygon
        double area = 0, length = 0;
        for (int i = 0, j = 3; i < 4; j = i++)
        {
            area += (double)box[j].X * box[i].Y - (double)box[i].X * box[j].Y;
            length += Math.Sqrt(Math.Pow((double)box[i].X - box[j].X, 2) + Math.Pow((double)box[i].Y - box[j].Y, 2));
        }
        area = Math.Abs(area) / 2;
        double distance = area * UnclipRatio / length;
        var path = box.Select(p => ((long)p.X, (long)p.Y)).ToList();
        return ClipperOffset(path, distance);
    }

    private static long CRound(double v) => v < 0 ? (long)(v - 0.5) : (long)(v + 0.5);

    /// <summary>Clipper 6.4.2 ClipperOffset, one closed polygon, round joins (the union afterwards changes no
    /// point of a convex outline)</summary>
    internal static List<(long X, long Y)> ClipperOffset(List<(long X, long Y)> input, double delta)
    {
        // AddPath: duplicate points out
        var path = new List<(long X, long Y)>();
        int highI = input.Count - 1;
        while (highI > 0 && input[0] == input[highI]) highI--;
        if (highI < 0) return new();
        path.Add(input[0]);
        for (int i = 1; i <= highI; i++)
            if (path[^1] != input[i]) path.Add(input[i]);
        if (path.Count < 3) return new();
        // FixOrientations
        double a = 0;
        for (int i = 0, j = path.Count - 1; i < path.Count; j = i++)
            a += ((double)path[j].X + path[i].X) * ((double)path[j].Y - path[i].Y);
        if (-a * 0.5 < 0) path.Reverse();
        if (delta > -1e-20 && delta < 1e-20) return path;
        const double pi = 3.141592653589793238, twoPi = pi * 2, arcTol = 0.25;
        double y = arcTol > Math.Abs(delta) * 0.25 ? Math.Abs(delta) * 0.25 : arcTol;
        double steps = pi / Math.Acos(1 - y / Math.Abs(delta));
        if (steps > Math.Abs(delta) * pi) steps = Math.Abs(delta) * pi;
        double sin = Math.Sin(twoPi / steps), cos = Math.Cos(twoPi / steps);
        double stepsPerRad = steps / twoPi;
        if (delta < 0.0) sin = -sin;
        int len = path.Count;
        var normals = new (double X, double Y)[len];
        static (double, double) UnitNormal((long X, long Y) p1, (long X, long Y) p2)
        {
            if (p2.X == p1.X && p2.Y == p1.Y) return (0, 0);
            double dx = p2.X - p1.X, dy = p2.Y - p1.Y;
            double f = 1 * 1.0 / Math.Sqrt(dx * dx + dy * dy);
            dx *= f; dy *= f;
            return (dy, -dx);
        }
        for (int j = 0; j < len - 1; j++) normals[j] = UnitNormal(path[j], path[j + 1]);
        normals[len - 1] = UnitNormal(path[len - 1], path[0]);
        var dest = new List<(long X, long Y)>();
        int k = len - 1;
        for (int j = 0; j < len; j++)
        {
            double sinA = normals[k].X * normals[j].Y - normals[j].X * normals[k].Y;
            if (Math.Abs(sinA * delta) < 1.0)
            {
                double cosA0 = normals[k].X * normals[j].X + normals[j].Y * normals[k].Y;
                if (cosA0 > 0)
                {
                    dest.Add((CRound(path[j].X + normals[k].X * delta), CRound(path[j].Y + normals[k].Y * delta)));
                    continue;
                }
            }
            else if (sinA > 1.0) sinA = 1.0;
            else if (sinA < -1.0) sinA = -1.0;
            if (sinA * delta < 0)
            {
                dest.Add((CRound(path[j].X + normals[k].X * delta), CRound(path[j].Y + normals[k].Y * delta)));
                dest.Add(path[j]);
                dest.Add((CRound(path[j].X + normals[j].X * delta), CRound(path[j].Y + normals[j].Y * delta)));
            }
            else
            {
                // DoRound
                double ang = Math.Atan2(sinA, normals[k].X * normals[j].X + normals[k].Y * normals[j].Y);
                int st = Math.Max((int)CRound(stepsPerRad * Math.Abs(ang)), 1);
                double X = normals[k].X, Y = normals[k].Y, X2;
                for (int i = 0; i < st; i++)
                {
                    dest.Add((CRound(path[j].X + X * delta), CRound(path[j].Y + Y * delta)));
                    X2 = X;
                    X = X * cos - sin * Y;
                    Y = X2 * sin + Y * cos;
                }
                dest.Add((CRound(path[j].X + normals[j].X * delta), CRound(path[j].Y + normals[j].Y * delta)));
            }
            k = j;
        }
        return dest;
    }

    private static float[,] OrderPointsClockwise(int[,] pts)
    {
        var idx = Py.Sorted(Enumerable.Range(0, 4), i => pts[i, 0]);
        var left = Py.Sorted(idx.Take(2), i => pts[i, 1]);
        var right = Py.Sorted(idx.Skip(2), i => pts[i, 1]);
        int tl = left[0], bl = left[1], tr = right[0], br = right[1];
        var rect = new float[4, 2];
        foreach (var (row, i) in new[] { (0, tl), (1, tr), (2, br), (3, bl) })
        {
            rect[row, 0] = pts[i, 0];
            rect[row, 1] = pts[i, 1];
        }
        return rect;
    }

    private static List<float[,]> SortedBoxes(List<float[,]> boxes)
    {
        var b = Py.Sorted(boxes, x => new Key2(x[0, 1], x[0, 0]));
        for (int i = 0; i < b.Count - 1; i++)
            for (int j = i; j >= 0; j--)
            {
                if (Math.Abs(b[j + 1][0, 1] - b[j][0, 1]) < 10 && b[j + 1][0, 0] < b[j][0, 0])
                    (b[j], b[j + 1]) = (b[j + 1], b[j]);
                else break;
            }
        return b;
    }

    // ── recognition: get_crop_img_list + text_rec ──
    /// <summary>the text in each box (x0, y0, x1, y1) of the RGB image, with its confidence</summary>
    public List<(string Text, double Score)> Recognise(Mat img, IList<double[]> boxes)
    {
        var crops = boxes.Select(b => Crop(img, new[]
        {
            new Point2f((float)b[0], (float)b[1]), new Point2f((float)b[2], (float)b[1]),
            new Point2f((float)b[2], (float)b[3]), new Point2f((float)b[0], (float)b[3]),
        })).ToList();
        try
        {
            return TextRec(crops);
        }
        finally
        {
            foreach (var c in crops) c.Dispose();
        }
    }

    private static Mat Crop(Mat img, Point2f[] points)
    {
        int cw = (int)Math.Max(NormF(points[0].X - points[1].X, points[0].Y - points[1].Y), NormF(points[2].X - points[3].X, points[2].Y - points[3].Y));
        int ch = (int)Math.Max(NormF(points[0].X - points[3].X, points[0].Y - points[3].Y), NormF(points[1].X - points[2].X, points[1].Y - points[2].Y));
        var std = new[] { new Point2f(0, 0), new Point2f(cw, 0), new Point2f(cw, ch), new Point2f(0, ch) };
        using var m = Cv2.GetPerspectiveTransform(points, std);
        var dst = new Mat();
        Cv2.WarpPerspective(img, dst, m, new Size(cw, ch), InterpolationFlags.Cubic, BorderTypes.Replicate);
        if (dst.Rows * 1.0 / dst.Cols >= 1.5)
        {
            var r = new Mat();
            Cv2.Rotate(dst, r, RotateFlags.Rotate90Counterclockwise);
            dst.Dispose();
            dst = r;
        }
        return dst;
    }

    private List<(string, double)> TextRec(List<Mat> imgs)
    {
        int n = imgs.Count;
        var ratios = imgs.Select(m => m.Cols / (double)m.Rows).ToArray();
        var indices = Py.Sorted(Enumerable.Range(0, n), i => ratios[i]);
        var res = new (string, double)[n];
        for (int i = 0; i < n; i++) res[i] = ("", 0.0);
        for (int beg = 0; beg < n; beg += RecBatch)
        {
            int end = Math.Min(n, beg + RecBatch);
            double maxRatio = RecW / (double)RecH;
            for (int k = beg; k < end; k++)
            {
                var m = imgs[indices[k]];
                maxRatio = Math.Max(maxRatio, m.Cols * 1.0 / m.Rows);
            }
            int imgW = (int)(RecH * maxRatio);
            var batch = new DenseTensor<float>(new[] { end - beg, 3, RecH, imgW });
            for (int k = beg; k < end; k++) ResizeNorm(imgs[indices[k]], imgW, batch, k - beg);
            float[] preds;
            int t, c;
            lock (_lock)
            {
                using var results = _rec.Run(new[] { NamedOnnxValue.CreateFromTensor(_recInput, batch) });
                var tensor = results.First().AsTensor<float>();
                t = tensor.Dimensions[1]; c = tensor.Dimensions[2];
                preds = tensor.ToArray();
            }
            for (int b = 0; b < end - beg; b++) res[indices[beg + b]] = Decode(preds, b, t, c);
        }
        return res.ToList();
    }

    private static void ResizeNorm(Mat img, int imgW, DenseTensor<float> batch, int b)
    {
        double ratio = img.Cols / (double)img.Rows;
        int resizedW = Math.Ceiling(RecH * ratio) > imgW ? imgW : (int)Math.Ceiling(RecH * ratio);
        using var resized = new Mat();
        Cv2.Resize(img, resized, new Size(resizedW, RecH));
        var idx = resized.GetGenericIndexer<Vec3b>();
        for (int y = 0; y < RecH; y++)
            for (int x = 0; x < resizedW; x++)
            {
                var v = idx[y, x];
                for (int ch = 0; ch < 3; ch++)
                {
                    float f = v[ch] / 255f;
                    f -= 0.5f;
                    f /= 0.5f;
                    batch[b, ch, y, x] = f;
                }
            }
    }

    /// <summary>CTC: the best character per step, repeats and blanks dropped; the score is the mean probability</summary>
    private (string, double) Decode(float[] preds, int b, int t, int c)
    {
        var sb = new System.Text.StringBuilder();
        var conf = new List<float>();
        int prev = -1;
        for (int s = 0; s < t; s++)
        {
            int off = (b * t + s) * c;
            int best = 0;
            float bp = preds[off];
            for (int k = 1; k < c; k++)
                if (preds[off + k] > bp) { bp = preds[off + k]; best = k; }
            bool keep = (s == 0 || best != prev) && best != 0;
            prev = best;
            if (!keep) continue;
            sb.Append(_chars[best]);
            conf.Add(bp);
        }
        if (conf.Count == 0) conf.Add(0);
        return (sb.ToString(), MeanF(conf.ToArray()));
    }

    /// <summary>numpy.mean of a float32 array (pairwise sum in float32)</summary>
    private static double MeanF(float[] a) => PairF(a, 0, a.Length) / a.Length;

    private static float PairF(float[] a, int s, int n)
    {
        if (n < 8)
        {
            float r = 0f;
            for (int i = 0; i < n; i++) r += a[s + i];
            return r;
        }
        if (n <= 128)
        {
            float r0 = a[s], r1 = a[s + 1], r2 = a[s + 2], r3 = a[s + 3], r4 = a[s + 4], r5 = a[s + 5], r6 = a[s + 6], r7 = a[s + 7];
            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                r0 += a[s + i]; r1 += a[s + i + 1]; r2 += a[s + i + 2]; r3 += a[s + i + 3];
                r4 += a[s + i + 4]; r5 += a[s + i + 5]; r6 += a[s + i + 6]; r7 += a[s + i + 7];
            }
            float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) res += a[s + i];
            return res;
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairF(a, s, n2) + PairF(a, s + n2, n - n2);
    }
}
