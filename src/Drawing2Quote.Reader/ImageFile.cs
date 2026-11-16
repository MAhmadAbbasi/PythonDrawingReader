using System.Buffers.Binary;
using OpenCvSharp;

namespace Drawing2Quote.Reader;

/// <summary>Pictures (JPG, PNG, TIFF, BMP, GIF, WEBP): their pages as RGB images (turned upright as the camera
/// recorded it) and the resolution they were scanned at, as Pillow reports it (info["dpi"])</summary>
internal static class ImageFile
{
    /// <summary>(width, height), or null when it isn't a picture that can be read</summary>
    public static (int W, int H)? Size(byte[] data)
    {
        var frames = Frames(data, firstOnly: true);
        if (frames.Count == 0) return null;
        var m = frames[0];
        var size = (m.Cols, m.Rows);
        foreach (var f in frames) f.Dispose();
        return size;
    }

    /// <summary>every page (a multi-page TIFF) as an RGB image</summary>
    public static List<Mat> Frames(byte[] data, bool firstOnly = false)
    {
        var out_ = new List<Mat>();
        bool tiff = data.Length >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 42) || (data[0] == 'M' && data[1] == 'M' && data[3] == 42));
        try
        {
            if (tiff && !firstOnly)
            {
                // a multi-page TIFF: every page (OpenCV reads several pages from a file only)
                var tmp = Path.Combine(Path.GetTempPath(), $"d2q-{Guid.NewGuid():N}.tif");
                try
                {
                    File.WriteAllBytes(tmp, data);
                    if (Cv2.ImReadMulti(tmp, out Mat[] pages, ImreadModes.Color) && pages.Length > 0)
                        foreach (var p in pages)
                        {
                            out_.Add(ToRgb(p));
                            p.Dispose();
                        }
                }
                finally
                {
                    try { File.Delete(tmp); } catch { }
                }
            }
            if (out_.Count == 0)
            {
                using var bgr = Cv2.ImDecode(data, ImreadModes.Color);
                if (!bgr.Empty()) out_.Add(ToRgb(bgr));
            }
        }
        catch (OpenCVException)
        {
        }
        if (out_.Count == 0) out_.AddRange(MuPdfFrames(data, firstOnly));
        return out_;
    }

    private static Mat ToRgb(Mat bgr)
    {
        var rgb = new Mat();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);
        return rgb;
    }

    /// <summary>formats OpenCV doesn't read (GIF): MuPDF's image loader</summary>
    private static List<Mat> MuPdfFrames(byte[] data, bool firstOnly)
    {
        var out_ = new List<Mat>();
        try
        {
            lock (PdfDoc.Lock)
            {
                using var pix = new MuPDF.NET.Pixmap(data);
                using var rgb = pix.N >= 3 && pix.Alpha == 0 ? null : new MuPDF.NET.Pixmap(MuPDF.NET.Colorspace.Rgb, pix);
                var p = rgb ?? pix;
                int w = p.Width, h = p.Height, n = p.N, stride = p.Stride;
                var samples = p.Samples;
                var mat = new Mat(h, w, MatType.CV_8UC3);
                var idx = mat.GetGenericIndexer<Vec3b>();
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int o = y * stride + x * n;
                        idx[y, x] = new Vec3b(samples[o], samples[o + 1], samples[o + 2]);
                    }
                out_.Add(mat);
            }
        }
        catch
        {
        }
        return out_;
    }

    /// <summary>Pillow's info["dpi"][0] (0 when the file doesn't say)</summary>
    public static double Dpi(byte[] data)
    {
        try
        {
            if (data.Length > 3 && data[0] == 0xff && data[1] == 0xd8) return JpegDpi(data);
            if (data.Length > 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') return PngDpi(data);
            if (data.Length > 8 && ((data[0] == 'I' && data[1] == 'I') || (data[0] == 'M' && data[1] == 'M'))) return TiffDpi(data, 0, data.Length, tiffFile: true);
            if (data.Length > 54 && data[0] == 'B' && data[1] == 'M') return BmpDpi(data);
        }
        catch
        {
        }
        return 0.0;
    }

    private static double JpegDpi(byte[] d)
    {
        double? dpi = null;
        byte[]? exif = null;
        int i = 2;
        while (i + 4 <= d.Length)
        {
            if (d[i] != 0xff) break;
            int marker = d[i + 1];
            if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker == 0x01 || marker == 0xff) { i += marker == 0xff ? 1 : 2; continue; }
            if (marker == 0xda || marker == 0xd9) break;          // start of scan: the header is read
            int len = (d[i + 2] << 8) | d[i + 3];
            int s = i + 4, n = len - 2;
            if (n < 0 || s + n > d.Length) break;
            if (marker == 0xe0 && n >= 5 && d[s] == 'J' && d[s + 1] == 'F' && d[s + 2] == 'I' && d[s + 3] == 'F' && n >= 12)
            {
                int unit = d[s + 7];
                int dx = (d[s + 8] << 8) | d[s + 9];
                if (unit == 1) dpi = dx;
                else if (unit == 2) dpi = dx * 2.54;
            }
            else if (marker == 0xe1 && n >= 6 && d[s] == 'E' && d[s + 1] == 'x' && d[s + 2] == 'i' && d[s + 3] == 'f' && d[s + 4] == 0 && d[s + 5] == 0)
            {
                exif ??= d[(s + 6)..(s + n)];
            }
            i = s + n;
        }
        if (dpi != null) return dpi.Value;
        if (exif == null) return 0.0;
        try
        {
            double v = TiffDpi(exif, 0, exif.Length, tiffFile: false);
            return v;
        }
        catch
        {
            return 72.0;
        }
    }

    /// <summary>the resolution in a TIFF structure (a TIFF file, or the EXIF block of a JPEG)</summary>
    private static double TiffDpi(byte[] d, int start, int length, bool tiffFile)
    {
        bool le = d[start] == 'I';
        ushort U16(int o) => le ? BinaryPrimitives.ReadUInt16LittleEndian(d.AsSpan(start + o)) : BinaryPrimitives.ReadUInt16BigEndian(d.AsSpan(start + o));
        uint U32(int o) => le ? BinaryPrimitives.ReadUInt32LittleEndian(d.AsSpan(start + o)) : BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(start + o));
        int ifd = (int)U32(4);
        int count = U16(ifd);
        double? xres = null;
        int? unit = null;
        for (int k = 0; k < count; k++)
        {
            int e = ifd + 2 + 12 * k;
            int tag = U16(e), type = U16(e + 2);
            if (tag == 0x011a)
            {
                int off = (int)U32(e + 8);
                uint num = U32(off), den = U32(off + 4);
                xres = type == 5 ? (den == 0 ? double.NaN : (double)num / den) : U32(e + 8);
            }
            else if (tag == 0x0128) unit = type == 3 ? U16(e + 8) : (int)U32(e + 8);
        }
        if (tiffFile)
        {
            double x = xres ?? 1;
            if (x == 0 || double.IsNaN(x)) return 0.0;
            if (unit == 2 || unit == null) return x;
            if (unit == 3) return x * 2.54;
            return 0.0;
        }
        // JPEG EXIF: both tags needed, else 72
        if (xres == null || unit == null || double.IsNaN(xres.Value)) return 72.0;
        return unit == 3 ? xres.Value * 2.54 : xres.Value;
    }

    private static double PngDpi(byte[] d)
    {
        int i = 8;
        while (i + 8 <= d.Length)
        {
            int len = (int)BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(i));
            var type = System.Text.Encoding.ASCII.GetString(d, i + 4, 4);
            if (type == "pHYs" && len >= 9)
            {
                uint px = BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(i + 8));
                int unit = d[i + 16];
                return unit == 1 ? px * 0.0254 : 0.0;
            }
            if (type == "IDAT" || type == "IEND") break;
            i += 12 + len;
        }
        return 0.0;
    }

    private static double BmpDpi(byte[] d)
    {
        int header = BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(14));
        if (header < 40) return 0.0;                    // (the old core header has no resolution)
        int ppm = BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(14 + 24));
        return ppm / 39.3701;
    }
}
