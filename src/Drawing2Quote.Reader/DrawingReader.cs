namespace Drawing2Quote.Reader;

/// <summary>
/// Reads an electrical drawing - DXF, a vector PDF from CAD, a printed PDF (OCR), a scanned PDF or a picture - into
/// floors -> areas -> electrical point counts. No AI or cloud service: MuPDF, NetTopologySuite, OpenCV and the
/// RapidOCR models on ONNX Runtime, with the same rules as the Python Drawing2Quote reader.
/// </summary>
public static class DrawingReader
{
    public static readonly string NotElectrical =
        "Not an electrical drawing: it has no electrical legend or points (an architectural layout). " +
        "The solution is suggested from the rooms and areas found only - lights, switches and sockets are " +
        "estimated per room.";

    public static string SupportedMessage => Validator.SupportedMessage;

    /// <summary>A picture is a floor plan when something of one was read in it: a room name, a floor title, symbols,
    /// or (names too small to read in a blurred photo) several rooms traced between walls. A screenshot or a table
    /// has none of these - at most a space or two between its lines.</summary>
    private static bool LooksLikePlan(IList<Sheet> sheets)
    {
        foreach (var s in sheets)
        {
            if (s.Points.Count > 0 || s.FloorTitles.Count > 0 || s.Views.Count > 0 || !s.DefaultFloor.StartsWith("Page ")) return true;
            if (s.Rooms.Any(r => r.Kind != null && r.Kind != "other")) return true;
            if (s.Rooms.Count(r => r.Poly != null) >= 4) return true;
        }
        return false;
    }

    /// <summary>The drawing's project info, floors with their areas and point counts, legend, warnings and a step log.
    /// Throws RejectedFileException when the file can't be read (with the reason).</summary>
    public static Extraction Extract(string fileName, byte[] data, SymbolMapper? mapper = null) =>
        ExtractWithSheets(fileName, data, mapper).Extraction;

    /// <summary>Extract(), plus the rooms and points as read on each sheet (for Overlay.Draw).
    /// pages: PDF page numbers to read (1-based; all when null)</summary>
    public static (Extraction Extraction, List<Sheet> Sheets) ExtractWithSheets(string fileName, byte[] data, SymbolMapper? mapper = null,
                                                                                IList<int>? pages = null)
    {
        var kind = Validator.Check(fileName, data);
        mapper ??= new SymbolMapper();
        var log = new List<string>();
        var warnings = new List<string>();
        Dictionary<string, string> project;
        List<Sheet> sheets;
        List<LegendRow> legend;
        try
        {
            if (kind == "pdf")
                (project, sheets, legend) = PdfReader.Read(data, mapper, log, warnings, pages);
            else if (kind is "scan" or "image")
            {
                // traced into a drawing first; the overlay is drawn on that traced drawing
                byte[] traced;
                (project, sheets, legend, traced) = RasterReader.Read(data, kind, mapper, log, warnings, pages);
                foreach (var s in sheets) s.Debug.Pdf ??= traced;
            }
            else
                (project, sheets, legend) = DxfReader.Read(data, mapper, log, warnings);
        }
        catch (RejectedFileException)
        {
            throw;
        }
        catch (Exception exc)
        {
            throw new RejectedFileException($"The drawing could not be read: {exc.Message}");
        }
        if (kind is "scan" or "image" && !LooksLikePlan(sheets))
            throw new RejectedFileException("This picture doesn't look like a floor plan: no room names, floor titles or electrical " +
                                            "symbols were found in it. Send a clear photo or scan of the drawing.");
        var floors = Common.Assemble(sheets, log, warnings);
        log.Add("Step 6/7 - points outside rooms are in \"Unassigned\"; every area has a confidence score");
        var sums = Common.Totals(floors);
        // every drawing found on the sheets (floor plans, area plans, elevations, sections ...): nothing left out unsaid
        var sections = new List<Section>();
        foreach (var s in sheets)
            foreach (var v in s.Debug.Views)
            {
                var floor = floors.FirstOrDefault(f => f.Name == v.Name);
                sections.Add(new Section
                {
                    Page = v.Page, Name = v.Name, Kind = v.Kind,
                    Rooms = floor != null ? floor.Areas.Count(a => a.Name != "Unassigned") : 0,
                    AreaSqft = floor?.AreaSqft,
                });
            }
        var others = sections.Where(v => v.Kind is not ("plan" or "area")).Select(v => $"{v.Name ?? "None"} (page {v.Page})").ToList();
        if (others.Count > 0)
            warnings.Add("Also on the sheets, not floor plans (no rooms taken from them): " + string.Join(", ", others) + ".");
        // an architectural layout (no electrical legend, no points): the solution comes from the rooms alone
        bool electrical = legend.Count > 0 || sums.Points > 0;
        if (!electrical)
        {
            // (the "no legend found" note would only repeat it; repeated shapes stay listed - symbols drawn without a
            // legend can be told apart there)
            warnings.RemoveAll(w => w.Contains("no legend found"));
            warnings.Insert(0, NotElectrical);
        }
        return (new Extraction
        {
            FileName = fileName, FileType = kind, Project = project, Floors = floors, Legend = legend, Totals = sums,
            Sections = sections, Electrical = electrical, Warnings = warnings, Log = log, PointTypes = mapper.TypesList.ToList(),
        }, sheets);
    }
}
