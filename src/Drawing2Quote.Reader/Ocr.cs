using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using OpenCvSharp;

namespace Drawing2Quote.Reader;

/// <summary>A word read by OCR: text, confidence, pixel box (x0, y0, x1, y1)</summary>
public readonly record struct OcrWord(string Text, double Conf, double X0, double Y0, double X1, double Y1);

/// <summary>
/// Text for PDFs whose words became lines (printed to PDF through a printer driver, e.g. "PScript5 / Distiller")
/// and for pictures: the page is rendered as an image and read with the RapidOCR models on ONNX Runtime - offline,
/// no cloud service.
///
/// Reading is slow (about half a minute per A3 sheet), so results are cached on disk by the page's content: the
/// same drawing is only read once. The cache is the Python reader's (same folder, keys and format), so a drawing
/// either of them has read is not read again. Set D2Q_OCR_CACHE to choose the cache folder.
/// </summary>
public static class Ocr
{
    public const int Dpi = 200;
    public const int Tile = 720, Overlap = 120;   // pixels; words cut by a tile edge are read whole in the next tile
    public const double MinConfidence = 0.6;

    private static readonly object EngineLock = new();
    private static RapidOcr? _engine;

    /// <summary>the folder with the OCR models (ch_PP-OCRv4_det_infer.onnx, ch_PP-OCRv4_rec_infer.onnx)</summary>
    public static string ModelDir = Environment.GetEnvironmentVariable("D2Q_MODELS") is { Length: > 0 } m
        ? m : Path.Combine(AppContext.BaseDirectory, "models");

    private static readonly AsyncLocal<Action<int, int>?> _progress = new();

    /// <summary>optional callback(page no, page count), called before a page is read (per reading: async-local)</summary>
    public static Action<int, int>? Progress
    {
        get => _progress.Value;
        set => _progress.Value = value;
    }

    public static bool Available() =>
        File.Exists(Path.Combine(ModelDir, "ch_PP-OCRv4_det_infer.onnx")) && File.Exists(Path.Combine(ModelDir, "ch_PP-OCRv4_rec_infer.onnx"));

    private static RapidOcr Engine()
    {
        lock (EngineLock) return _engine ??= new RapidOcr(ModelDir);
    }

    /// <summary>the OCR cache folder; null when it can't be made (e.g. an IIS app pool without a writable temp folder):
    /// the words are then read without a cache</summary>
    private static string? CacheDir()
    {
        var d = Environment.GetEnvironmentVariable("D2Q_OCR_CACHE") is { Length: > 0 } c ? c : Path.Combine(Path.GetTempPath(), "drawing2quote-ocr");
        try
        {
            Directory.CreateDirectory(d);
            return d;
        }
        catch (Exception exc) when (exc is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return null;
        }
    }

    internal static string Sha1(byte[] data) => Convert.ToHexString(SHA1.HashData(data)).ToLowerInvariant();

    internal static string Sha1(string s) => Sha1(Encoding.UTF8.GetBytes(s));

    private static List<OcrWord> ReadImage(Mat img)
    {
        var engine = Engine();
        // 1. find where the words are, tile by tile (small tiles: the detector shrinks big images and loses small text)
        var boxes = new List<double[]>();
        int h = img.Rows, w = img.Cols;
        for (int y = 0; y < h; y += Tile - Overlap)
            for (int x = 0; x < w; x += Tile - Overlap)
            {
                var roi = new OpenCvSharp.Rect(x, y, Math.Min(Tile, w - x), Math.Min(Tile, h - y));
                using var tile = new Mat(img, roi).Clone();
                if (InkShare(tile) < 0.002) continue;          // (nearly) empty paper
                foreach (var q in engine.Detect(tile))
                {
                    double qx0 = double.PositiveInfinity, qy0 = double.PositiveInfinity, qx1 = double.NegativeInfinity, qy1 = double.NegativeInfinity;
                    for (int p = 0; p < 4; p++)
                    {
                        double px = (double)q[p, 0] + x, py = (double)q[p, 1] + y;
                        qx0 = Math.Min(qx0, px); qy0 = Math.Min(qy0, py); qx1 = Math.Max(qx1, px); qy1 = Math.Max(qy1, py);
                    }
                    boxes.Add(new[] { qx0, qy0, qx1, qy1 });
                }
            }
        if (boxes.Count == 0) return new List<OcrWord>();
        // 2. tiles overlap: the same word found twice, or a long line cut by a tile edge -> one box
        boxes = MergeBoxes(boxes);
        // 3. read every word once, all in one batch (reading is the slow part)
        var texts = engine.Recognise(img, boxes);
        return texts.Select((t, i) => new OcrWord(t.Text, t.Score, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])).ToList();
    }

    /// <summary>share of the tile's pixels darker than 200 in some channel</summary>
    private static double InkShare(Mat tile)
    {
        var idx = tile.GetGenericIndexer<Vec3b>();
        long dark = 0;
        for (int y = 0; y < tile.Rows; y++)
            for (int x = 0; x < tile.Cols; x++)
            {
                var v = idx[y, x];
                if (Math.Min(v.Item0, Math.Min(v.Item1, v.Item2)) < 200) dark++;
            }
        return (double)dark / (tile.Rows * (long)tile.Cols);
    }

    /// <summary>Boxes on the same text line that overlap become one (union); a box inside a bigger one is dropped</summary>
    internal static List<double[]> MergeBoxes(List<double[]> boxes)
    {
        var b = Py.Sorted(boxes, r => r[0]);
        bool merged = true;
        while (merged)
        {
            merged = false;
            var out_ = new List<double[]>();
            foreach (var r in b)
            {
                bool broke = false;
                foreach (var k in out_)
                {
                    double ix = Math.Min(r[2], k[2]) - Math.Max(r[0], k[0]);
                    double iy = Math.Min(r[3], k[3]) - Math.Max(r[1], k[1]);
                    double hr = r[3] - r[1], hk = k[3] - k[1];
                    if (ix > 0 && iy > 0.6 * Math.Min(hr, hk) && Math.Max(hr, hk) < 1.6 * Math.Min(hr, hk))
                    {
                        k[0] = Math.Min(k[0], r[0]); k[1] = Math.Min(k[1], r[1]); k[2] = Math.Max(k[2], r[2]); k[3] = Math.Max(k[3], r[3]);
                        merged = true;
                        broke = true;
                        break;
                    }
                    double area = Math.Max((r[2] - r[0]) * (r[3] - r[1]), 1e-6);
                    if (ix > 0 && iy > 0 && ix * iy > 0.7 * area)
                    {
                        merged = true;                     // mostly inside a box already kept
                        broke = true;
                        break;
                    }
                }
                if (!broke) out_.Add((double[])r.Clone());
            }
            b = out_;
        }
        return b;
    }

    /// <summary>Tiles overlap, so a word can be read twice or in pieces: of overlapping boxes keep the longest text</summary>
    private static List<OcrWord> Dedupe(IEnumerable<OcrWord> words)
    {
        var sorted = Py.Sorted(words, w => new Key2(-w.Text.Trim().Length, -w.Conf));
        var kept = new List<OcrWord>();
        foreach (var w in sorted)
        {
            double area = Math.Max((w.X1 - w.X0) * (w.Y1 - w.Y0), 1e-6);
            bool dup = false;
            foreach (var k in kept)
            {
                double ix = Math.Max(0.0, Math.Min(w.X1, k.X1) - Math.Max(w.X0, k.X0));
                double iy = Math.Max(0.0, Math.Min(w.Y1, k.Y1) - Math.Max(w.Y0, k.Y0));
                if (ix * iy > 0.5 * Math.Min(area, (k.X1 - k.X0) * (k.Y1 - k.Y0)))
                {
                    dup = true;
                    break;
                }
            }
            if (!dup) kept.Add(w);
        }
        return kept;
    }

    /// <summary>the words of an image (text, confidence, pixel box), read once: image() gives the RGB image when needed</summary>
    private static List<OcrWord> CachedWords(string key, Func<Mat> image)
    {
        var dir = CacheDir();
        if (dir == null)
        {
            using var picture = image();
            return Dedupe(ReadImage(picture).Where(w => w.Conf >= MinConfidence && w.Text.Trim().Length > 0));
        }
        var cache = Path.Combine(dir, Sha1(key) + ".json");
        if (File.Exists(cache))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(cache, Encoding.UTF8));
                var words = new List<OcrWord>();
                foreach (var w in doc.RootElement.EnumerateArray())
                {
                    var b = w[2].EnumerateArray().Select(v => v.GetDouble()).ToArray();
                    words.Add(new OcrWord(w[0].GetString() ?? "", w[1].GetDouble(), b[0], b[1], b[2], b[3]));
                }
                return words;
            }
            catch
            {
                // a damaged cache file: read again
            }
        }
        List<OcrWord> read;
        using (var img = image())
            read = Dedupe(ReadImage(img).Where(w => w.Conf >= MinConfidence && w.Text.Trim().Length > 0));
        // the cache only saves time: a file that can't be written (disk full, locked by the Python reader) doesn't fail the
        // reading. It is written whole beside and then moved in, so neither reader ever sees half a file.
        var part = $"{cache}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.part";
        try
        {
            using (var fs = File.Create(part))
            using (var jw = new Utf8JsonWriter(fs))
            {
                jw.WriteStartArray();
                foreach (var w in read)
                {
                    jw.WriteStartArray();
                    jw.WriteStringValue(w.Text);
                    jw.WriteNumberValue(w.Conf);
                    jw.WriteStartArray();
                    jw.WriteNumberValue(w.X0); jw.WriteNumberValue(w.Y0); jw.WriteNumberValue(w.X1); jw.WriteNumberValue(w.Y1);
                    jw.WriteEndArray();
                    jw.WriteEndArray();
                }
                jw.WriteEndArray();
            }
            File.Move(part, cache, overwrite: true);
        }
        catch (Exception exc) when (exc is IOException or UnauthorizedAccessException)
        {
            try { File.Delete(part); } catch (Exception) { /* left for the next time */ }
        }
        return read;
    }

    /// <summary>the image's pixels as numpy's img.tobytes() (RGB, row by row)</summary>
    internal static byte[] Bytes(Mat img)
    {
        var data = new byte[img.Rows * img.Cols * img.Channels()];
        if (img.IsContinuous()) System.Runtime.InteropServices.Marshal.Copy(img.Data, data, 0, data.Length);
        else
        {
            int rowLen = img.Cols * img.Channels();
            for (int y = 0; y < img.Rows; y++)
                System.Runtime.InteropServices.Marshal.Copy(img.Ptr(y), data, y * rowLen, rowLen);
        }
        return data;
    }

    /// <summary>the words of a picture or scanned page (RGB image): (text, confidence, pixel box)</summary>
    public static List<OcrWord> ImageWords(Mat img)
    {
        var key = $"img1:{Sha1(Bytes(img))}:({img.Rows}, {img.Cols}, {img.Channels()}):{Tile}";
        return CachedWords(key, () => img.Clone());
    }

    /// <summary>OCR words (pixel boxes) as text lines in PDF points, misreadings of drawing words corrected</summary>
    public static List<(TextItem T, PRect R)> ToLines(IEnumerable<OcrWord> words, double ptPerPx)
    {
        var out_ = new List<(TextItem, PRect)>();
        foreach (var w in words)
        {
            var r = new PRect(w.X0 * ptPerPx, w.Y0 * ptPerPx, w.X1 * ptPerPx, w.Y1 * ptPerPx);
            out_.Add((new TextItem(Correct(w.Text.Trim()), (r.X0 + r.X1) / 2, (r.Y0 + r.Y1) / 2, r.Height * 0.8), r));
        }
        return out_;
    }

    // words of room names, legends and titles: small or blurred text is often read a letter or two off
    // ("LIVIRG LOLNGE", "DRAWNG", "MOINZNKITCIEN") and words run together ("CARPARKING")
    private static readonly string[] Vocabulary = Py.Sorted(
        Py.Split(@"
KITCHEN LOUNGE LIVING DRAWING DINING BEDROOM BED ROOM MASTER GUEST BATH BATHROOM TOILET DRESS DRESSING WASHROOM
POWDER PASSAGE LOBBY FOYER ENTRANCE CORRIDOR GALLERY STAIR STAIRS STAIRCASE STORE STORAGE GARAGE PORCH PARKING
LAUNDRY UTILITY SERVANT QUARTER QUARTERS DRIVER STUDY OFFICE LIBRARY PRAYER LAWN GARDEN COURTYARD PATIO
TERRACE BALCONY VERANDAH PANTRY FAMILY SITTING HALL MODERN ISLAND DOUBLE HEIGHT FRONT BACK AREA KIDS LIFT
LIGHT LIGHTS SWITCH SOCKET POWER CEILING EXHAUST POINT BOARD DISTRIBUTION LEGEND SYMBOL SYMBOLS DESCRIPTION
FLOOR GROUND FIRST SECOND THIRD BASEMENT PLAN LAYOUT ELECTRICAL CURTAIN CAR
").Distinct(), w => w.Length, reverse: true).ToArray();

    private static readonly HashSet<string> VocabSet = new(Vocabulary);

    private static double Close(string a, string b) => Py.SequenceRatio(a, b);

    /// <summary>drawing words misread by OCR put right; anything else is left as it was</summary>
    public static string Correct(string text)
    {
        var out_ = new List<string>();
        foreach (var token in text.Split(' '))
        {
            var word = token.ToUpperInvariant();
            if (word.Length < 4 || !Py.IsAlpha(word) || VocabSet.Contains(word))
            {
                out_.Add(token);
                continue;
            }
            // the closest drawing word (ties: the alphabetically last, as Python's max over (ratio, word))
            double bestRatio = 0;
            string bestWord = "";
            bool any = false;
            foreach (var v in Vocabulary)
            {
                if (Math.Abs(v.Length - word.Length) > 2) continue;
                double r = Close(word, v);
                if (!any || r > bestRatio || (r == bestRatio && string.CompareOrdinal(v, bestWord) > 0))
                {
                    bestRatio = r; bestWord = v; any = true;
                }
            }
            if (bestRatio >= 0.8)
            {
                out_.Add(bestWord);
                continue;
            }
            if (word.Length >= 7)                  // several words run together: the drawing words found in it
            {
                var pieces = new List<string>();
                int i = 0, known = 0;
                var rest = new StringBuilder();
                while (i < word.Length)
                {
                    // a word spelt right first (the longest), else one a letter off with the same first and last letters
                    (string V, int N)? hit = null;
                    foreach (var v in Vocabulary)
                        if (i + v.Length <= word.Length && string.CompareOrdinal(word, i, v, 0, v.Length) == 0) { hit = (v, v.Length); break; }
                    if (hit == null)
                    {
                        foreach (var v in Vocabulary)
                        {
                            if (v.Length < 5) continue;
                            foreach (var n in new[] { v.Length, v.Length - 1, v.Length + 1 })
                            {
                                if (i + n > word.Length) continue;
                                var part = word.Substring(i, n);
                                if (part.Length == n && part[0] == v[0] && part[^1] == v[^1] && Close(part, v) >= 0.84)
                                {
                                    hit = (v, n);
                                    break;
                                }
                            }
                            if (hit != null) break;
                        }
                    }
                    if (hit is { } h)
                    {
                        if (rest.Length > 0)
                        {
                            pieces.Add(rest.ToString());        // letters that are no drawing word stay as they were
                            rest.Clear();
                        }
                        pieces.Add(h.V);
                        known += h.N;
                        i += h.N;
                    }
                    else
                    {
                        rest.Append(word[i]);
                        i++;
                    }
                }
                if (rest.Length > 0) pieces.Add(rest.ToString());
                if (known >= 0.4 * word.Length && pieces.Any(p => p.Length >= 4 && VocabSet.Contains(p)))
                {
                    out_.Add(string.Join(" ", pieces));
                    continue;
                }
            }
            out_.Add(token);
        }
        return string.Join(" ", out_);
    }

    /// <summary>Text lines read from the page image, in the coordinates of the page as shown (rotation applied)</summary>
    internal static List<(TextItem T, PRect R)> PageLines(PdfPage page, string docHash)
    {
        var words = CachedWords($"v2:{docHash}:{page.Number}:{Dpi}:{Tile}", () => page.RenderRgb(Dpi));
        return ToLines(words, 72.0 / Dpi);
    }
}
