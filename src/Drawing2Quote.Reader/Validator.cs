namespace Drawing2Quote.Reader;

/// <summary>Upload checks: DXF, a PDF of a CAD drawing (exported, or printed - then read with OCR), or a scanned
/// PDF or picture of a drawing (read from the image)</summary>
public static class Validator
{
    public const int MaxMb = 200;     // keep the same in the POS: DrawingService.MaxBytes, web.config, drawingService.ts
    public const long MaxBytes = MaxMb * 1024L * 1024;

    // keep the same wording as READ_QUALITY in the POS frontend (drawingService.ts)
    public static readonly string SupportedMessage =
        "Best: DXF or CAD-exported PDF (exact) · Good: CAD PDF without text (OCR, slower) · Fair: scan or clear " +
        "JPG/PNG (check counts) · Low: blurry photo/screenshot (rooms & sizes estimated). " +
        $"Cannot read: DWG (save as DXF), locked PDF, HEIC, under 300 px, over {MaxMb} MB. " +
        "Points need a legend; send WhatsApp images as Document.";

    private static readonly HashSet<string> ImageExt = new() { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp" };

    private static readonly byte[][] ImageMagic =
    {
        new byte[] { 0xff, 0xd8, 0xff }, new byte[] { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a },
        "II*\0"u8.ToArray(), "MM\0*"u8.ToArray(), "BM"u8.ToArray(), "GIF8"u8.ToArray(), "RIFF"u8.ToArray(),
    };

    /// <summary>Path(name).suffix.lower() as Python gives it</summary>
    internal static string Suffix(string? fileName)
    {
        var name = Path.GetFileName(fileName ?? "");
        int i = name.LastIndexOf('.');
        return 0 < i && i < name.Length - 1 ? name[i..].ToLowerInvariant() : "";
    }

    private static bool StartsWith(byte[] data, byte[] prefix) => data.Length >= prefix.Length && data.AsSpan(0, prefix.Length).SequenceEqual(prefix);

    /// <summary>"dxf", "pdf" (CAD PDF), "scan" (scanned PDF) or "image" - throws RejectedFileException with the reason otherwise</summary>
    public static string Check(string? fileName, byte[] data)
    {
        var ext = Suffix(fileName);
        if (data == null || data.Length == 0) throw new RejectedFileException("The file is empty.");
        if (data.Length > MaxBytes)
            throw new RejectedFileException($"The file is {Py.Fixed(data.Length / 1024.0 / 1024.0, 1)} MB - the maximum is {MaxMb} MB.");
        if (ext == ".dwg" || StartsWith(data, "AC10"u8.ToArray()) || StartsWith(data, "AC1."u8.ToArray()))
            throw new RejectedFileException("DWG files can't be read. In AutoCAD use Save As -> DXF and upload the DXF file.");
        if (ext is ".heic" or ".heif")
            throw new RejectedFileException("HEIC photos can't be read. On the phone choose \"Most Compatible\" (JPG) or send it as JPG.");
        if (ImageExt.Contains(ext) || ImageMagic.Any(m => StartsWith(data, m)))
        {
            CheckImage(data);
            return "image";
        }
        if (ext == ".pdf" || StartsWith(data, "%PDF-"u8.ToArray())) return CheckPdf(data);
        if (ext == ".dxf")
        {
            var head = System.Text.Encoding.Latin1.GetString(data, 0, Math.Min(4096, data.Length));
            if (!head.Contains("SECTION") && !StartsWith(data, "AutoCAD Binary DXF"u8.ToArray()))
                throw new RejectedFileException("This file is not a valid DXF drawing. Save it from AutoCAD as DXF and upload it again.");
            return "dxf";
        }
        throw new RejectedFileException($"\"{(ext.Length > 0 ? ext : "this")}\" files are not supported. " + SupportedMessage);
    }

    private static void NeedOcr()
    {
        if (!Ocr.Available())
            throw new RejectedFileException("Scanned drawings and pictures need the OCR reader (its model files are missing " +
                                            "from the models folder). Or upload the DXF (in AutoCAD: Save As -> DXF).");
    }

    private static void CheckImage(byte[] data)
    {
        (int W, int H)? size;
        try
        {
            size = ImageFile.Size(data);
        }
        catch
        {
            size = null;
        }
        if (size == null)
            throw new RejectedFileException("This picture is damaged or not a picture format that can be read (use JPG, PNG or TIFF).");
        var (w, h) = size.Value;
        if (Math.Min(w, h) < 300)
            throw new RejectedFileException($"This picture is too small ({w} x {h} pixels) to read the room names and symbols. " +
                                            "Scan or photograph the drawing at a higher resolution (at least 2000 pixels wide).");
        NeedOcr();
    }

    private static string CheckPdf(byte[] data)
    {
        PdfDoc doc;
        try
        {
            doc = new PdfDoc(data);
        }
        catch
        {
            throw new RejectedFileException("This PDF is damaged and can't be opened.");
        }
        using (doc)
        {
            if (doc.NeedsPass || doc.IsEncrypted)
                throw new RejectedFileException("This PDF is password-protected. Upload a copy without a password.");
            int count = doc.PageCount;
            if (count == 0) throw new RejectedFileException("This PDF has no pages.");
            bool hasText = false, hasVectors = false;
            for (int i = 0; i < count; i++)
            {
                var page = doc[i];
                if (page.PlainText().Trim().Length > 0) hasText = true;
                // count line pieces, not paths: CAD exports often put thousands of lines in one path
                if (!hasVectors && page.GetDrawings().Sum(d => d.Items.Count) >= 20) hasVectors = true;
                if (hasText && hasVectors) return "pdf";
            }
            if (!hasVectors)
            {
                // scanned: the pages are pictures (a searchable scan has a text layer too) - read from the image
                if (!Enumerable.Range(0, count).Any(i => doc.PageImageCount(i) > 0))
                    throw new RejectedFileException("This PDF has no drawing in it (no lines and no scanned pages).");
                NeedOcr();
                return "scan";
            }
            if (!hasText)
            {
                if (Ocr.Available()) return "pdf";     // printed to PDF: the words are read from the page image (OCR)
                throw new RejectedFileException("This PDF has the drawing lines but no selectable text - it was printed to PDF, so every " +
                                                "word became lines and the room names and legend can't be read. Install the OCR reader " +
                                                "(the models folder), or upload the DXF (in AutoCAD: Save As -> DXF).");
            }
            return "pdf";
        }
    }
}
