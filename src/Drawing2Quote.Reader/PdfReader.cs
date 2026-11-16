using System.Text.RegularExpressions;
using NetTopologySuite.Geometries;
using OpenCvSharp;

namespace Drawing2Quote.Reader;

/// <summary>A drawing found on a sheet: its box, its name and what it is (plan, area, elevation, section ...)</summary>
internal sealed class View
{
    public PRect Box;
    public string? Name;
    public string Kind = "other";
    public bool Titled;
}

/// <summary>What one page gave: its legend rows, and the sheet (rooms, points) when it was read</summary>
internal sealed class PageResult
{
    public int Page;
    public bool Legend;
    public List<KeyValuePair<string, LegendRow>> Rows = new();
    public Sheet? Sheet;
    public List<string> Log = new();
    public List<string> Warnings = new();
    public (List<(Symbol Sym, string Kind)> Shapes, Dictionary<string, string> Texts)? OwnLegend;
    public double LegendShare;
}

/// <summary>
/// Vector PDF exported from CAD: each page is a floor (or several floors with their titles).
///
/// Per page:
///   1. text lines (room names, dimensions, legend, title block) and the drawing broken into strokes
///   2. legend: each description row and the symbol drawn to its left (a shape, a shape with a letter, or a short
///      text like "AC")
///   3. symbols: small strokes that touch are grouped into symbols and matched to the closest legend symbol by
///      shape, at any rotation (see PdfGeometry)
///   4. rooms: the enclosed space around each room name, traced from the wall lines with door openings closed;
///      else the smallest drawn rectangle around the name; else only the name's position
///   5. areas: room dimensions written on the plan ("14'x12'"), else outline x drawing scale (from "SCALE 1:100",
///      or worked out from the rooms that have dimensions written)
/// </summary>
public static class PdfReader
{
    private static readonly Regex Scale = new(@"1\s*:\s*(\d+)", Py.RX);
    // 1/8" = 1'-0"   (inches on paper per foot)
    private static readonly Regex ScaleImperial = new(@"(\d+)\s*/\s*(\d+)\s*(?:""|''|”|″)?\s*=\s*1\s*(?:'|’|′)", Py.RX);

    /// <summary>the drawing scale as 1:N, from "SCALE 1:100" or "SCALE: 1/8"=1'-0""</summary>
    private static int? ScaleOf(IEnumerable<TextItem> texts)
    {
        foreach (var t in texts)
        {
            if (!Common.Norm(t.Text).Contains("SCALE")) continue;
            var m = ScaleImperial.Match(t.Text);
            if (m.Success)
            {
                int num = Py.ParseInt(m.Groups[1].Value), den = Py.ParseInt(m.Groups[2].Value);
                if (num != 0 && den != 0) return Py.RoundInt(12.0 * den / num);
            }
            m = Scale.Match(t.Text);
            if (m.Success) return Py.ParseInt(m.Groups[1].Value);
        }
        return null;
    }

    /// <summary>text lines in the coordinates of the page as shown (MuPDF gives them unrotated)</summary>
    internal static List<(TextItem T, PRect R)> Lines(PdfPage page)
    {
        var m = page.RotationMatrix;
        var out_ = new List<(TextItem, PRect)>();
        foreach (var line in page.TextLines())
        {
            // spans far apart on one line are separate texts (two room names side by side)
            var groups = new List<List<(string Text, PRect BBox, double Size)>>();
            foreach (var span in line)
            {
                if (span.Text.Trim().Length == 0) continue;
                if (groups.Count > 0 && span.BBox.X0 - groups[^1][^1].BBox.X1 < 1.5 * span.Size) groups[^1].Add(span);
                else groups.Add(new() { span });
            }
            foreach (var spans in groups)
            {
                var text = string.Join(" ", spans.Select(s => s.Text.Trim()));
                var r = (spans[0].BBox | spans[^1].BBox).Transform(m);
                out_.Add((new TextItem(text, (r.X0 + r.X1) / 2, (r.Y0 + r.Y1) / 2, spans.Max(s => s.Size)), r));
            }
        }
        return out_;
    }

    private static readonly HashSet<string> NameWords = new()
    {
        "ROOM", "AREA", "HALL", "QUARTER", "QUARTERS", "AND", "GUEST", "STAFF", "DRESS", "FAMILY", "ATTACHED",
        "MAIN", "UPPER", "LOWER", "CHILDREN", "SON", "DAUGHTER", "PARENTS", "GRAND", "WALK", "IN", "CLOSET",
    };
    private static readonly Regex NoteWords = new(@"^(?:PVC|CONDUIT|CONDUI|CABLE|LPDB\w*|SQMM|WIRE|FROM|TO)+", Py.RX);

    /// <summary>a word that can be part of a room name on another line (2 letters only if a room word itself: "TV")</summary>
    private static bool IsWord(string text)
    {
        var n = Common.Norm(text);
        if (!(n.Length <= 20 && Py.FullMatch("[A-Z ]+", n) != null)) return false;
        // a room word itself, or a word that goes with one ("GUEST" / "ROOM"); not a nearby note ("PVC CONDUIT")
        return Common.RoomType(n) != null || Py.Split(n).Any(NameWords.Contains);
    }

    /// <summary>circuit tags read together with the name ("1S/EY1STAFF QUARTER" -> "STAFF QUARTER")</summary>
    private static string CleanName(string name)
    {
        var words = new List<string>();
        foreach (var w0 in Py.Split(name))
        {
            var w = w0;
            var stripped = NoteWords.Replace(w, "");     // a cable note read together with the name ("PVCCONDUITPASSAGE")
            if (stripped != w && Common.RoomType(stripped) != null) w = stripped;
            w = w.Trim(',', '.', ';', ':', '(', ')', '[', ']', '{', '}');
            if (w.Length == 0) continue;
            if (Py.IsDigit(w))                           // "BED ROOM 2"
            {
                words.Add(w);
                continue;
            }
            if (Py.Search(@"[\d/]", w))
            {
                var tail = Regex.Replace(w.ToUpperInvariant(), "^.*[^A-Z]", "", Py.RX);   // the letters after the tag: "1S/EY1STAFF" -> "STAFF"
                if (tail.Length >= 3 && Py.IsAlpha(tail)) words.Add(tail);
                continue;
            }
            words.Add(w);
        }
        var s = string.Join(" ", words);
        return s.Length > 0 ? s : name;
    }

    // the most a room of a kind can measure (sq ft) - a traced outline beyond it ran out through an opening. Open
    // spaces (lawn, garage / porch, balcony / terrace) and halls have no limit.
    private static readonly Dictionary<string, double> RoomMaxSqft = new()
    {
        ["bath"] = 250, ["dressing"] = 300, ["kitchen"] = 600, ["store"] = 400, ["laundry"] = 300, ["stairs"] = 500,
        ["bedroom"] = 1000, ["servant"] = 600, ["study"] = 900, ["dining"] = 1200,
    };

    // the least (sq ft): smaller is a sofa, a bed or a stair tread traced instead of the room
    private static readonly Dictionary<string, double> RoomMinSqft = new()
    {
        ["bedroom"] = 50, ["lounge"] = 60, ["dining"] = 50, ["kitchen"] = 25, ["study"] = 40, ["servant"] = 30, ["bath"] = 12,
        ["stairs"] = 10, ["store"] = 10, ["laundry"] = 10, ["dressing"] = 10,
    };

    private static readonly HashSet<string> ServiceRooms = new() { "bath", "dressing", "store", "kitchen", "laundry", "stairs" };

    // a void in the floor ("LOOK BELOW" over a double-height hall): not part of any room on this floor
    private static readonly Regex OpenBelow = new(@"\b(LOOK BELOW|OPEN TO BELOW|OPEN TO SKY)\b", Py.RX);

    // an address in the title block ("PLOT NO 290-J, STREET # 16, BLOCK A, GARDEN SCHEME B17") - not a room name
    private static readonly Regex Address = new(@"\b(SCHEME|SOCIETY|PHASE|SECTOR|STREET|PLOT NO|HOUSING|TOWN ?SHIP|COLONY)\b", Py.RX);

    private static bool TitleText(string text)
    {
        var n = Common.Norm(text);
        return Address.IsMatch(n) || Common.IsTitleField(n);
    }

    internal sealed record Label(string Name, string Kind, PRect Rect, double H);

    /// <summary>(name, room type, text box, text height); a name on two lines ("MASTER" / "BED ROOM") is one label.
    /// skip: texts that are symbols ("AC", the "TV" inside a TV point symbol)</summary>
    private static List<Label> RoomLabels(List<(TextItem T, PRect R)> lines, PRect? legendRect, ISet<int> skip)
    {
        var dims = lines.Where(l => Common.ParseDimensions(l.T.Text) != null).Select(l => l.R).ToList();

        bool Sized(PRect r) =>   // a room size written right under it ("MUMTY" / "11'-7" X 7'-2"")
            dims.Any(d => -0.5 * r.Height <= d.Y0 - r.Y1 && d.Y0 - r.Y1 < 2.5 * r.Height && Math.Abs(d.CX - r.CX) < Math.Max(d.Width, r.Width));

        var free = new List<(TextItem T, PRect R)>();
        for (int k = 0; k < lines.Count; k++)
        {
            var (t, r) = lines[k];
            if (skip.Contains(k) || (legendRect is { } lr && lr.Intersects(r))) continue;
            if (!(Common.FloorName(t.Text) == null || (Common.RoomType(t.Text) != null && Sized(r)))) continue;
            if (TitleText(t.Text) || Py.Search(@"\d{4,}", t.Text)) continue;    // (an account code, a phone number - not a room name)
            free.Add((t, r));
        }
        var used = new HashSet<int>();
        var labels = new List<Label>();
        for (int i = 0; i < free.Count; i++)
        {
            var (t, r) = free[i];
            var kind = Common.RoomType(t.Text);
            if (kind == null || used.Contains(i)) continue;
            var parts = new List<(double Y, string Text)> { (r.Y0, t.Text.Trim()) };
            var rect = r;
            used.Add(i);
            bool grew = true;
            while (grew)                     // words stacked right above / below the name
            {
                grew = false;
                for (int j = 0; j < free.Count; j++)
                {
                    var (o, orr) = free[j];
                    if (used.Contains(j) || !IsWord(o.Text) || Common.ParseDimensions(o.Text) != null) continue;
                    double h = Math.Max(rect.Height / Math.Max(parts.Count, 1), orr.Height);
                    bool stacked = Math.Min(Math.Abs(orr.Y0 - rect.Y1), Math.Abs(rect.Y0 - orr.Y1)) < 0.6 * h;
                    bool centred = Math.Abs(orr.CX - rect.CX) < Math.Max(rect.Width, orr.Width) / 2 + h;
                    if (stacked && centred)
                    {
                        parts.Add((orr.Y0, o.Text.Trim()));
                        rect |= orr;
                        used.Add(j);
                        grew = true;
                    }
                }
            }
            var ordered = parts.OrderBy(p => p.Y).ThenBy(p => p.Text, StringComparer.Ordinal).Select(p => p.Text);
            var name = CleanName(string.Join(" ", ordered));
            labels.Add(new Label(name, Common.RoomType(name) ?? kind, rect, t.H));
        }
        return labels;
    }

    private static (double, double, double, double) T4(PRect r) => (r.X0, r.Y0, r.X1, r.Y1);

    /// <summary>legend symbols as shapes, as short texts, the legend's area and its rows for the log</summary>
    private static (List<(Symbol Sym, string Kind)> Shapes, Dictionary<string, string> Texts, PRect? Rect, List<KeyValuePair<string, LegendRow>> Rows)
        Legend(List<(TextItem T, PRect R)> lines, List<Stroke> items, SymbolMapper mapper)
    {
        var shapes = new List<(Symbol, string)>();
        var texts = new Dictionary<string, string>();
        var rowsOut = new List<KeyValuePair<string, LegendRow>>();
        void SetRow(string key, LegendRow row)
        {
            if (!rowsOut.Any(kv => kv.Key == key)) rowsOut.Add(new(key, row));
        }
        PRect? legendRect = null;
        foreach (var (title, tr) in lines.Where(l => Common.IsLegendTitle(Common.Norm(l.T.Text))).ToList())
        {
            // the rows under the title: texts starting near the title's left edge, one after another
            var below = Py.Sorted(lines.Where(l => l.R.Y0 > tr.Y0 + 1 && tr.X0 - 4 * tr.Height < l.R.X0 && l.R.X0 < tr.X0 + 6 * tr.Height), l => l.R.Y0);
            var rows = new List<(TextItem T, PRect R)>();
            var last = tr;
            foreach (var (t, r) in below)
            {
                if (r.Y0 - last.Y1 > 3 * r.Height) break;
                rows.Add((t, r));
                last = r;
            }
            if (rows.Count == 0) continue;
            var area = tr;
            foreach (var (_, r) in rows) area |= r;
            foreach (var (t, r) in rows)
            {
                var kind = mapper.Classify(t.Text);
                if (string.IsNullOrEmpty(kind)) continue;
                double hh = r.Height;
                var strip = (r.X0 - 6 * hh, r.Y0 - 1.5 * hh, r.X0 + 1, r.Y1 + 1.5 * hh);
                var candidates = PdfGeometry.FindSymbols(items.Where(s => s.Inside(strip)).ToList(), 4 * hh, 0.15 * hh)
                                            .Where(s => s.BBox.X1 <= r.X0 + 1 && Math.Abs(s.Y - (r.Y0 + r.Y1) / 2) < hh).ToList();
                var shortT = lines.Where(l => l.R.X1 <= r.X0 + 1 && r.X0 - l.R.X1 < 6 * hh && Math.Abs(l.R.Y0 - r.Y0) < hh / 2
                                              && l.T.Text.Trim().Length <= 4).ToList();
                if (candidates.Count > 0)
                {
                    var sym = Py.MaxBy(candidates, s => s.BBox.X1);
                    sym.Text = InnerText(sym, shortT);
                    shapes.Add((sym, kind));
                    area |= PRect.From(sym.BBox);
                    SetRow(t.Text, new LegendRow { Symbol = "shape" + (sym.Text.Length > 0 ? $" with \"{sym.Text}\"" : ""), Description = t.Text, Type = kind });
                }
                else if (shortT.Count > 0)
                {
                    texts[Common.Norm(shortT[0].T.Text)] = kind;
                    area |= shortT[0].R;
                    SetRow(t.Text, new LegendRow { Symbol = shortT[0].T.Text, Description = t.Text, Type = kind });
                }
            }
            legendRect = legendRect is { } lr ? lr | area : area;
        }
        var table = LegendTable(lines, items, mapper);
        if (table.Count > 0)                        // a table legend: the row reading above saw its cells wrongly
        {
            shapes = new();
            texts = new();
            rowsOut = new();
            legendRect = null;
        }
        foreach (var (sym, kind, desc, area) in table)
        {
            shapes.Add((sym, kind));
            SetRow(desc, new LegendRow { Symbol = "shape", Description = desc, Type = kind });
            legendRect = legendRect is { } lr ? lr | area : area;
        }
        if (legendRect is { } l2) legendRect = l2.Add(-2, -2, 2, 2);
        if (shapes.Count >= 4)                      // bits much smaller than the other legend symbols (a letter, a tick) match junk
        {
            double typical = Py.Median(shapes.Select(s => s.Item1.Size));
            shapes = shapes.Where(s => s.Item1.Size >= 0.45 * typical).ToList();
        }
        return (shapes, texts, legendRect, rowsOut);
    }

    /// <summary>A legend drawn as a table: a SYMBOL(S) column and a DESCRIPTION column. Every symbol drawn in the
    /// symbol column of a row counts as that row's type (a row can show several: normal / smart / dimmer switch).</summary>
    private static List<(Symbol Sym, string Kind, string Desc, PRect Area)> LegendTable(List<(TextItem T, PRect R)> lines, List<Stroke> items, SymbolMapper mapper)
    {
        var out_ = new List<(Symbol, string, string, PRect)>();
        // (column headers: SYMBOL | DESCRIPTION, or PLAN | NAME - a header only counts when it has its partner on the
        // same row)
        var heads = lines.Where(l => Py.FullMatch("SYMBOLS?|PLAN", Common.Norm(l.T.Text)) != null).Select(l => l.R).ToList();
        var descs = lines.Where(l => Common.Norm(l.T.Text).StartsWith("DESCRIPTION") || Common.Norm(l.T.Text) == "NAME").Select(l => l.R).ToList();

        PRect? Pair(int si, string side)
        {
            // the DESCRIPTION header of this SYMBOL header's table: on the same header row, on that side, with no
            // other SYMBOL header between them (that one is another table's)
            var sr = heads[si];
            double cy = (sr.Y0 + sr.Y1) / 2;
            var found = new List<(double D, PRect R)>();
            foreach (var dr in descs)
            {
                if (Math.Abs((dr.Y0 + dr.Y1) / 2 - cy) >= sr.Height) continue;
                var (a, b) = side == "right" ? (sr.X1, dr.X0) : (dr.X1, sr.X0);
                if (0 < b - a && b - a < 60 * sr.Height
                    && !heads.Where((h, hi) => hi != si && Math.Abs((h.Y0 + h.Y1) / 2 - cy) < sr.Height && a < (h.X0 + h.X1) / 2 && (h.X0 + h.X1) / 2 < b).Any())
                    found.Add((b - a, dr));
            }
            return found.Count > 0 ? Py.MinBy(found, f => f.D).R : null;
        }

        // the symbol column left of the descriptions (SYMBOL | DESCRIPTION) or right of them (DESCRIPTION | HEIGHT |
        // SYMBOL): the way that gives more tables their own description column (one sheet draws its tables alike)
        var bySide = new Dictionary<string, List<(PRect Sr, PRect Dr)>>();
        foreach (var side0 in new[] { "right", "left" })
        {
            var l = new List<(PRect, PRect)>();
            for (int si = 0; si < heads.Count; si++)
                if (Pair(si, side0) is { } dr) l.Add((heads[si], dr));
            bySide[side0] = l;
        }
        var side = bySide["left"].Count > bySide["right"].Count ? "left" : "right";
        // (a table ends at the header row of the table below it - not at a legend title, which can head a group of
        // rows inside the table: "LIGHTING LEGEND")
        var stops = descs;
        foreach (var (sr, dr) in bySide[side])
        {
            (double A, double B) col = side == "right" ? (sr.X1, (dr.X0 + dr.X1) / 2) : (dr.X0 - 1.5 * dr.Width, dr.X1);
            double top = Math.Max(sr.Y1, dr.Y1);
            var xSpan = (A: Math.Min(col.A, sr.X0), B: Math.Max(col.B, sr.X1));
            double end = stops.Where(r => r.Y0 > top + 1 && r.X1 > xSpan.A && r.X0 < xSpan.B).Select(r => r.Y0).DefaultIfEmpty(double.PositiveInfinity).Min();
            var cells = Py.Sorted(lines.Where(l => top - 1 < l.R.Y0 && l.R.Y0 < end && col.A < l.R.X0 && l.R.X0 < col.B && l.R != sr && l.R != dr), l => l.R.Y0);
            var rows = new List<List<(TextItem T, PRect R)>>();
            double lastY = sr.Y1;
            foreach (var (t, r) in cells)
            {
                if (rows.Count > 0 && Math.Abs((r.Y0 + r.Y1) / 2 - (rows[^1][0].R.Y0 + rows[^1][0].R.Y1) / 2) < r.Height / 2)
                {
                    rows[^1].Add((t, r));            // more words on the same row
                    continue;
                }
                if (r.Y0 - lastY > 8 * Math.Max(r.Height, 2)) break;   // end of the table
                rows.Add(new() { (t, r) });
                lastY = r.Y1;
            }
            if (rows.Count == 0) continue;
            if (side == "left")
            {
                out_.AddRange(SymbolColumn(rows, sr, dr, items, mapper));
                continue;
            }
            var centres = rows.Select(row => (row[0].R.Y0 + row[0].R.Y1) / 2).ToList();
            double tx0 = sr.X0 - 2 * sr.Width, tx1 = rows.SelectMany(row => row).Max(x => x.R.X0);
            double ty0 = sr.Y1 - 1, ty1 = rows[^1].Max(x => x.R.Y1) + 12;
            var tableItems = items.Where(s => s.BBox.X0 >= tx0 && s.BBox.X1 <= tx1 && s.BBox.Y0 >= ty0 && s.BBox.Y1 <= ty1).ToList();
            double pitch = centres.Count > 1 ? Py.Median(centres.Zip(centres.Skip(1), (a, b) => b - a)) : 3 * rows[0][0].R.Height;
            double left = sr.X0 - 1.5 * sr.Width;
            for (int ri = 0; ri < rows.Count; ri++)
            {
                var row = rows[ri];
                double y = centres[ri];
                var desc = string.Join(" ", Py.Sorted(row, x => x.R.X0).Select(x => x.T.Text));
                var kind = mapper.Classify(desc);
                if (string.IsNullOrEmpty(kind)) continue;
                double xEnd = row.Min(x => x.R.X0) - 0.5;
                var strip = (left, y - 0.5 * pitch, xEnd, y + 0.5 * pitch);
                // symbols can be wider than the row is high (a distribution board); table borders are long lines
                var found = PdfGeometry.FindSymbols(tableItems.Where(s => s.Inside(strip)).ToList(), 0.9 * (xEnd - left), 0.04 * pitch,
                                                    maxLine: 0.8 * pitch);
                if (found.Count == 0) continue;
                double biggest = found.Max(s => s.Size);
                foreach (var sym in found)
                    if (sym.Size >= 0.4 * biggest)
                        out_.Add((sym, kind, desc, sr | dr | PRect.From(strip) | row[^1].R));
            }
        }
        return out_;
    }

    /// <summary>A table with its symbol column right of the descriptions (DESCRIPTION | HEIGHT | SYMBOL): rows can be
    /// much taller than their text (a fan drawn big), so the symbols are found in the whole column and each goes to
    /// the nearest description.</summary>
    private static List<(Symbol, string, string, PRect)> SymbolColumn(List<List<(TextItem T, PRect R)>> rows, PRect sr, PRect dr,
                                                                      List<Stroke> items, SymbolMapper mapper)
    {
        var joined = new List<List<(TextItem T, PRect R)>>();   // a name on two lines in its cell
        foreach (var row in rows)
        {
            if (joined.Count > 0 && string.IsNullOrEmpty(mapper.Classify(string.Join(" ", joined[^1].Select(x => x.T.Text))))
                && row[0].R.Y0 - joined[^1].Max(x => x.R.Y1) < row[0].R.Height)
                joined[^1] = joined[^1].Concat(row).ToList();
            else joined.Add(row);
        }
        var centres = joined.Select(row => (row.Min(x => x.R.Y0) + row.Max(x => x.R.Y1)) / 2).ToList();
        double pitch = centres.Count > 1 ? Py.Median(centres.Zip(centres.Skip(1), (a, b) => b - a)) : 3 * joined[0][0].R.Height;
        double cx = (sr.X0 + sr.X1) / 2;
        double tx0 = cx - 1.4 * sr.Width, tx1 = cx + 1.4 * sr.Width;
        double ty0 = sr.Y1 - 1, ty1 = joined[^1].Max(x => x.R.Y1) + pitch;
        var column = items.Where(s => s.BBox.X0 >= tx0 && s.BBox.X1 <= tx1 && s.BBox.Y0 >= ty0 && s.BBox.Y1 <= ty1).ToList();
        var found = PdfGeometry.FindSymbols(column, 0.9 * (tx1 - tx0), 0.04 * pitch)
                               .Where(s => Math.Abs(s.X - cx) < 0.75 * sr.Width).ToList();      // (not a column border line)
        var byRow = new Dictionary<int, List<Symbol>>();
        var rowOrder = new List<int>();
        foreach (var s in found)
        {
            int i = Py.MinBy(Enumerable.Range(0, centres.Count), k => Math.Abs(centres[k] - s.Y));
            if (Math.Abs(centres[i] - s.Y) < 1.5 * pitch)
            {
                if (!byRow.TryGetValue(i, out var l)) { byRow[i] = l = new List<Symbol>(); rowOrder.Add(i); }
                l.Add(s);
            }
        }
        var out_ = new List<(Symbol, string, string, PRect)>();
        foreach (var i in rowOrder)
        {
            var syms = byRow[i];
            var desc = string.Join(" ", joined[i].OrderBy(x => (long)Py.Round(x.R.Y0)).ThenBy(x => x.R.X0).Select(x => x.T.Text));
            var kind = mapper.Classify(desc);
            if (string.IsNullOrEmpty(kind)) continue;
            if (syms.Count > 3)                     // one symbol drawn as many loose pieces (a chandelier's ring of beads)
            {
                double x0 = syms.Min(s => s.BBox.X0), y0 = syms.Min(s => s.BBox.Y0);
                double x1 = syms.Max(s => s.BBox.X1), y1 = syms.Max(s => s.BBox.Y1);
                if (Math.Max(x1 - x0, y1 - y0) < 1.5 * pitch)
                {
                    // (described whole: FindSymbols would split it again at its dots, as switches stacked on a wall)
                    var whole = PdfGeometry.Describe(column.Where(s => s.Inside((x0 - 1, y0 - 1, x1 + 1, y1 + 1))).ToList());
                    if (whole != null) syms = new List<Symbol> { whole };
                }
            }
            // (a plain straight line - a cable, a strip light - cannot be told from wiring on the plans)
            syms = syms.Where(s => s.Filled || s.Curve > 0.05 || Enumerable.Range(0, 12).Max(k => s.Dirs[k] + s.Dirs[(k + 1) % 12]) < 0.97).ToList();
            if (syms.Count == 0) continue;
            double biggest = syms.Max(s => s.Size);
            var area = sr | dr | new PRect(tx0, centres[i] - 0.5 * pitch, tx1, centres[i] + 0.5 * pitch);
            foreach (var r in joined[i]) area |= r.R;
            foreach (var s in syms)
                if (s.Size >= 0.4 * biggest) out_.Add((s, kind, desc, area));
        }
        return out_;
    }

    // titles of the drawings on a sheet that are not floor plans (floor plans are named by FloorName)
    private static readonly (Regex Pattern, string Kind)[] ViewTitle =
    {
        (new Regex(@"\bELEVATIONS?\b", Py.RX), "elevation"),
        (new Regex(@"\bSECTIONS?\b(?! CUT)", Py.RX), "section"),
        (new Regex(@"\bAREA PLAN\b|\bAREA CHART\b", Py.RX), "area"),
        (new Regex(@"\bSITE PLAN\b|\bLOCATION PLAN\b|\bKEY PLAN\b", Py.RX), "site"),
        (new Regex(@"\bDETAILS?\b", Py.RX), "detail"),
    };
    internal static readonly HashSet<string> NotPlan = new() { "elevation", "section", "site", "detail" };

    /// <summary>(name, kind) when the text is the title of a drawing on the sheet: a floor plan ("GROUND FLOOR PLAN",
    /// kind "plan"), an elevation, a section, an area plan ...</summary>
    private static (string Name, string Kind)? ViewTitleOf(string text)
    {
        var n = Common.Norm(text);
        if (n.Length == 0 || Py.Split(n).Length > 7) return null;
        foreach (var (pattern, kind) in ViewTitle)
            if (pattern.IsMatch(n)) return (string.Join(" ", Py.Split(n).Select(Py.Capitalize)), kind);
        if (Common.FloorName(text) is { } floor) return (floor, "plan");
        return null;
    }

    /// <summary>The separate drawings on a sheet (several floor plans, an elevation, a section, an area plan side by
    /// side): blocks of drawing with clear paper between them, each named by the title written under (or in) it.</summary>
    internal static List<View> Views(List<Stroke> items, List<(TextItem T, PRect R)> lines, PRect pageRect, PRect? legendRect)
    {
        double W = pageRect.Width, H = pageRect.Height;
        double g = Math.Max(1.0, Math.Min(W, H) / 600);   // grid cell (points)
        int gw = (int)(W / g) + 2, gh = (int)(H / g) + 2;
        using var canvas = new Mat(gh, gw, MatType.CV_8UC1, Scalar.All(0));
        foreach (var s in items)
        {
            var (x0, y0, x1, y1) = s.BBox;
            if (x1 - x0 > 0.8 * W || y1 - y0 > 0.8 * H) continue;      // the sheet frame and title block borders
            if (legendRect is { } lr && s.Inside(T4(lr))) continue;
            var pts = s.Pts.Select(p => new OpenCvSharp.Point((int)Math.Round(p.X / g, MidpointRounding.ToEven), (int)Math.Round(p.Y / g, MidpointRounding.ToEven))).ToArray();
            Cv2.Polylines(canvas, new[] { pts }, false, Scalar.All(1), 1);
        }
        int gap = Math.Max(3, (int)(0.012 * Math.Min(W, H) / g));
        using var joined = new Mat();
        using (var kernel = Mat.Ones(gap, gap, MatType.CV_8UC1).ToMat())
            Cv2.Dilate(canvas, joined, kernel);
        using var lab = new Mat();
        using var stats = new Mat();
        using var cents = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(joined, lab, stats, cents, PixelConnectivity.Connectivity8);
        var boxes = new List<double[]>();
        for (int i = 1; i < n; i++)
        {
            int x = stats.At<int>(i, 0), y = stats.At<int>(i, 1), w = stats.At<int>(i, 2), h = stats.At<int>(i, 3);
            if (Math.Min(w, h) * g >= 0.12 * Math.Min(W, H))    // (a title block strip or a note is not a drawing)
                boxes.Add(new[] { x * g, y * g, (x + w) * g, (y + h) * g });
        }
        // drawings overlapping each other are one
        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i < boxes.Count && !merged; i++)
                for (int j = i + 1; j < boxes.Count; j++)
                {
                    var a = boxes[i]; var b = boxes[j];
                    if (a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3])
                    {
                        boxes[i] = new[] { Math.Min(a[0], b[0]), Math.Min(a[1], b[1]), Math.Max(a[2], b[2]), Math.Max(a[3], b[3]) };
                        boxes.RemoveAt(j);
                        merged = true;
                        break;
                    }
                }
        }
        var titles = new List<((string Name, string Kind) Vt, PRect R)>();
        foreach (var (t, r) in lines)
            if (ViewTitleOf(t.Text) is { } vt && !(legendRect is { } lr && lr.Intersects(r))) titles.Add((vt, r));
        // two drawings drawn close together join into one block: two titles side by side under it split it again
        var split = new List<double[]>();
        foreach (var b in boxes)
        {
            var under = titles.Select(x => x.R).Where(r => b[0] <= (r.X0 + r.X1) / 2 && (r.X0 + r.X1) / 2 <= b[2]
                                                           && -0.25 * (b[3] - b[1]) <= (r.Y0 + r.Y1) / 2 - b[3] && (r.Y0 + r.Y1) / 2 - b[3] <= 0.12 * H)
                              .Select(r => (r.X0 + r.X1) / 2).OrderBy(v => v).ToList();
            var xs = under.Where((x, i) => i == 0 || x - under[i - 1] > 0.25 * (b[2] - b[0])).ToList();
            var cuts = new List<double> { b[0] };
            for (int i = 0; i + 1 < xs.Count; i++) cuts.Add((xs[i] + xs[i + 1]) / 2);
            cuts.Add(b[2]);
            for (int i = 0; i + 1 < cuts.Count; i++) split.Add(new[] { cuts[i], b[1], cuts[i + 1], b[3] });
        }
        var views = split.Select(b => new View { Box = new PRect(b[0], b[1], b[2], b[3]) }).ToList();
        // titles: under the drawing (or inside it, near its foot), across its middle; the nearest such drawing
        foreach (var ((name, kind), r) in Py.Sorted(titles, x => -x.R.Height))
        {
            double cx = (r.X0 + r.X1) / 2, cy = (r.Y0 + r.Y1) / 2;
            (double D, View V)? best = null;
            foreach (var v in views)
            {
                var (x0, y0, x1, y1) = (v.Box.X0, v.Box.Y0, v.Box.X1, v.Box.Y1);
                if (v.Titled || !(x0 - 0.1 * (x1 - x0) <= cx && cx <= x1 + 0.1 * (x1 - x0))) continue;
                double below = cy - y1;                      // under its foot (negative: inside it)
                if (-0.25 * (y1 - y0) <= below && below <= 0.12 * H)
                {
                    double d = Math.Abs(below);
                    if (best == null || d < best.Value.D) best = (d, v);
                }
            }
            if (best == null)                               // a title written beside the drawing: the nearest one, close by
            {
                foreach (var v in views)
                {
                    var (x0, y0, x1, y1) = (v.Box.X0, v.Box.Y0, v.Box.X1, v.Box.Y1);
                    double d = Math.Pow(Math.Pow(Math.Max(Math.Max(x0 - cx, 0), cx - x1), 2) + Math.Pow(Math.Max(Math.Max(y0 - cy, 0), cy - y1), 2), 0.5);
                    if (!v.Titled && d <= 0.15 * W && (best == null || d < best.Value.D)) best = (d, v);
                }
            }
            if (best is { } bb)
            {
                bb.V.Name = name;
                bb.V.Kind = kind;
                bb.V.Titled = true;
            }
        }
        return views;
    }

    /// <summary>(type, description, where its symbol is drawn) for each legend row - for pictures, whose symbols are
    /// matched as images (RasterReader). A table legend (SYMBOL | DESCRIPTION) wins over plain rows.</summary>
    internal static List<(string Kind, string Desc, PRect Strip)> LegendRows(List<(TextItem T, PRect R)> lines, SymbolMapper mapper)
    {
        var table = new List<(string, string, PRect)>();
        var heads = lines.Where(l => Py.FullMatch("SYMBOLS?", Common.Norm(l.T.Text)) != null).ToList();
        var descs = lines.Where(l => Common.Norm(l.T.Text).StartsWith("DESCRIPTION")).ToList();
        foreach (var (sh, sr) in heads)
        {
            double cy = (sr.Y0 + sr.Y1) / 2;
            var right = descs.Where(d => d.R.X0 > sr.X1 && d.R.X0 - sr.X1 < 60 * sr.Height && Math.Abs((d.R.Y0 + d.R.Y1) / 2 - cy) < sr.Height).ToList();
            if (right.Count == 0) continue;
            var (dt, dr) = Py.MinBy(right, d => d.R.X0);
            var cells = Py.Sorted(lines.Where(l => l.R.Y0 > sr.Y1 && sr.X1 < l.R.X0 && l.R.X0 < (dr.X0 + dr.X1) / 2), l => l.R.Y0);
            var rows = new List<List<(TextItem T, PRect R)>>();
            double lastY = sr.Y1;
            foreach (var (t, r) in cells)
            {
                if (rows.Count > 0 && Math.Abs((r.Y0 + r.Y1) / 2 - (rows[^1][0].R.Y0 + rows[^1][0].R.Y1) / 2) < r.Height / 2)
                {
                    rows[^1].Add((t, r));
                    continue;
                }
                if (r.Y0 - lastY > 8 * Math.Max(r.Height, 2)) break;
                rows.Add(new() { (t, r) });
                lastY = r.Y1;
            }
            if (rows.Count == 0) continue;
            var centres = rows.Select(row => (row[0].R.Y0 + row[0].R.Y1) / 2).ToList();
            double pitch = centres.Count > 1 ? Py.Median(centres.Zip(centres.Skip(1), (a, b) => b - a)) : 3 * rows[0][0].R.Height;
            double left = sr.X0 - 1.5 * sr.Width;
            for (int ri = 0; ri < rows.Count; ri++)
            {
                var row = rows[ri];
                double y = centres[ri];
                var desc = string.Join(" ", Py.Sorted(row, x => x.R.X0).Select(x => x.T.Text));
                var kind = mapper.Classify(desc);
                if (!string.IsNullOrEmpty(kind))
                {
                    double xEnd = row.Min(x => x.R.X0) - 0.5;
                    table.Add((kind, desc, new PRect(left, y - 0.5 * pitch, xEnd, y + 0.5 * pitch)));
                }
            }
        }
        if (table.Count > 0) return table;
        var out_ = new List<(string, string, PRect)>();
        foreach (var (title, tr) in lines.Where(l => Common.IsLegendTitle(Common.Norm(l.T.Text))).ToList())
        {
            var below = Py.Sorted(lines.Where(l => l.R.Y0 > tr.Y0 + 1 && tr.X0 - 4 * tr.Height < l.R.X0 && l.R.X0 < tr.X0 + 6 * tr.Height), l => l.R.Y0);
            var rows = new List<(TextItem T, PRect R)>();
            var last = tr;
            foreach (var (t, r) in below)
            {
                if (r.Y0 - last.Y1 > 3 * r.Height) break;
                last = r;
                // a short text with a description to its right on the same line is a symbol ("AC"), not a row
                if (t.Text.Trim().Length <= 4 && lines.Any(o => !ReferenceEquals(o.T, t) && o.R.X0 > r.X1 && o.R.X0 - r.X1 < 8 * r.Height
                                                                && Math.Abs((o.R.Y0 + o.R.Y1) / 2 - (r.Y0 + r.Y1) / 2) < r.Height / 2))
                    continue;
                rows.Add((t, r));
            }
            // each row's symbol lies left of its text, up to halfway to the rows above and below
            var mids = rows.Select(x => (x.R.Y0 + x.R.Y1) / 2).ToList();
            for (int i = 0; i < rows.Count; i++)
            {
                var (t, r) = rows[i];
                var kind = mapper.Classify(t.Text);
                if (string.IsNullOrEmpty(kind)) continue;
                double hh = r.Height;
                double top = i > 0 ? (mids[i - 1] + mids[i]) / 2 : mids[i] - 1.5 * hh;
                double bottom = i + 1 < rows.Count ? (mids[i] + mids[i + 1]) / 2 : mids[i] + 1.5 * hh;
                out_.Add((kind, t.Text, new PRect(r.X0 - 6 * hh, Math.Max(top, mids[i] - 1.5 * hh), r.X0 - 0.5, Math.Min(bottom, mids[i] + 1.5 * hh))));
            }
        }
        return out_;
    }

    /// <summary>(type, how many) when the symbol is a blob of touching dots of a dot-drawn legend type</summary>
    private static (string Kind, int N)? Dots(Symbol sym, Dictionary<string, (string Kind, double One)> dotKinds)
    {
        if (!sym.Filled || !dotKinds.TryGetValue(sym.Color, out var dk)) return null;
        int n = Py.RoundInt(sym.FillArea / dk.One);
        return 1 <= n && n <= 12 && Math.Abs(sym.FillArea / dk.One - n) < 0.35 ? (dk.Kind, n) : null;
    }

    private static string InnerText(Symbol sym, List<(TextItem T, PRect R)> lines)
    {
        var (x0, y0, x1, y1) = sym.BBox;
        var inner = lines.Where(l => l.T.Text.Trim().Length <= 4 && x0 <= l.T.X && l.T.X <= x1 && y0 <= l.T.Y && l.T.Y <= y1).ToList();
        return inner.Count > 0 ? Common.Norm(inner[0].T.Text) : "";
    }

    private static readonly Regex LengthRx = new(@"\A(?:\s*(\d{1,3})\s*['′]\s*(?:-?\s*(\d{1,2})\s*([½¼¾])?\s*[""″]?)?\s*)\z", Py.RX);
    private static readonly Dictionary<string, double> Fraction = new() { ["½"] = 0.5, ["¼"] = 0.25, ["¾"] = 0.75 };

    /// <summary>(PDF points per foot, how many dimensions agreed) when the scale isn't written (N.T.S.): from the
    /// dimension strings - a length written on its dimension line ("9'", "2'-6\"", "7'-1½\""), the line as long as
    /// that many feet. null when there are too few, or they don't agree.</summary>
    private static (double Ft, int Agree)? FeetFromDimensions(List<(TextItem T, PRect R)> lines, List<Stroke> items)
    {
        var segs = items.Where(s => s.Pts.Length == 2 && !s.Filled).Select(s => (X0: s.Pts[0].X, Y0: s.Pts[0].Y, X1: s.Pts[^1].X, Y1: s.Pts[^1].Y)).ToList();
        if (segs.Count == 0) return null;
        var ratios = new List<double>();
        foreach (var (t, r) in lines)
        {
            var m = LengthRx.Match(t.Text);
            if (!m.Success) continue;
            double feet = Py.ParseInt(m.Groups[1].Value)
                          + ((m.Groups[2].Success ? Py.ParseInt(m.Groups[2].Value) : 0) + (m.Groups[3].Success && Fraction.TryGetValue(m.Groups[3].Value, out var fr) ? fr : 0)) / 12.0;
            if (!(1 <= feet && feet <= 200)) continue;
            double cx = (r.X0 + r.X1) / 2, cy = (r.Y0 + r.Y1) / 2;
            int best = -1;
            double bestOff = 0, bestLen = 0;
            for (int k = 0; k < segs.Count; k++)
            {
                var s = segs[k];
                double dx = s.X1 - s.X0, dy = s.Y1 - s.Y0, off, length;
                bool sel;
                if (r.Width >= r.Height)            // written along a horizontal line, just above / below it
                {
                    double h = r.Height;
                    sel = Math.Abs(dy) < 0.3 && Math.Min(s.X0, s.X1) <= cx && Math.Max(s.X0, s.X1) >= cx && Math.Abs(dx) >= r.Width && Math.Abs(s.Y0 - cy) < 1.5 * h;
                    off = Math.Abs(s.Y0 - cy);
                    length = Math.Abs(dx);
                }
                else                                // written up a vertical line
                {
                    double h = r.Width;
                    sel = Math.Abs(dx) < 0.3 && Math.Min(s.Y0, s.Y1) <= cy && Math.Max(s.Y0, s.Y1) >= cy && Math.Abs(dy) >= r.Height && Math.Abs(s.X0 - cx) < 1.5 * h;
                    off = Math.Abs(s.X0 - cx);
                    length = Math.Abs(dy);
                }
                if (sel && (best < 0 || off < bestOff)) { best = k; bestOff = off; bestLen = length; }
            }
            if (best >= 0) ratios.Add(bestLen / feet);
        }
        if (ratios.Count < 6) return null;
        double med = Py.Median(ratios);
        int agree = ratios.Count(x => Math.Abs(x / med - 1) < 0.08);
        return agree >= Math.Max(6, 0.5 * ratios.Count) ? (med, agree) : null;
    }

    /// <summary>PDF points per foot when the scale isn't written: room names are roughly 13 ft apart</summary>
    private static double FeetGuess(List<Label> labels, PdfPage page)
    {
        var pts = labels.Select(l => ((l.Rect.X0 + l.Rect.X1) / 2, (l.Rect.Y0 + l.Rect.Y1) / 2)).ToList();
        if (pts.Count >= 2)
        {
            var nearest = new List<double>();
            for (int i = 0; i < pts.Count; i++)
            {
                double best = double.PositiveInfinity;
                for (int j = 0; j < pts.Count; j++)
                {
                    if (i == j) continue;
                    double dx = pts[i].Item1 - pts[j].Item1, dy = pts[i].Item2 - pts[j].Item2;
                    best = Math.Min(best, Math.Sqrt(dx * dx + dy * dy));
                }
                nearest.Add(best);
            }
            return Py.Median(nearest) / 13;
        }
        return Math.Min(page.Rect.Width, page.Rect.Height) / 80;
    }

    private static bool LegendWord(string text)
    {
        var n = Common.Norm(text);
        return Common.IsLegendTitle(n) || n.StartsWith("SYMBOL") || n.StartsWith("DESCRIPTION") || n.Replace(" ", "").Contains("LEGEND");
    }

    private static List<PRect> WordBoxes(List<(TextItem T, PRect R)> lines) =>
        lines.Where(l => Common.Norm(l.T.Text).Replace(" ", "").Length >= 3).Select(l => l.R.Add(-0.5, -0.5, 0.5, 0.5)).ToList();

    /// <summary>indexes of the boxes lying within one of the word boxes (closed containment, as GEOS's "within" for boxes)</summary>
    private static HashSet<int> Within(List<PRect> words, IList<(double X0, double Y0, double X1, double Y1)> boxes)
    {
        var tree = new Geo.Tree(words.Select(w => Geo.Env(w.X0, w.Y0, w.X1, w.Y1)));
        var out_ = new HashSet<int>();
        for (int i = 0; i < boxes.Count; i++)
        {
            var b = boxes[i];
            foreach (int j in tree.Query(Geo.Env(b.X0, b.Y0, b.X1, b.Y1)))
            {
                var w = words[j];
                if (w.X0 <= b.X0 && b.X1 <= w.X1 && w.Y0 <= b.Y0 && b.Y1 <= w.Y1) { out_.Add(i); break; }
            }
        }
        return out_;
    }

    /// <summary>drawn paths lying inside a word read by OCR (3+ letters): the word's letters</summary>
    private static HashSet<int> WordPaths(List<Drawing> drawings, List<(TextItem T, PRect R)> lines, Mat2D matrix)
    {
        var words = WordBoxes(lines);
        if (words.Count == 0 || drawings.Count == 0) return new HashSet<int>();
        return Within(words, drawings.Select(d => T4(d.Rect.Transform(matrix))).ToList());
    }

    /// <summary>Strokes that are the outlines of words read by OCR (3+ letters; short tags like "S1" stay, so a symbol
    /// with a letter in it keeps its shape)</summary>
    private static List<Stroke> DropWordOutlines(List<Stroke> items, List<(TextItem T, PRect R)> lines)
    {
        var words = WordBoxes(lines);
        if (words.Count == 0 || items.Count == 0) return items;
        var drop = Within(words, items.Select(s => s.BBox).ToList());
        return items.Where((s, i) => !drop.Contains(i)).ToList();
    }

    private static readonly Regex RoomN = new(@"\AROOM \d+\z", Py.RX);

    private static bool AllClose(Pt a, Pt b) =>
        Math.Abs(a.X - b.X) <= 1e-08 + 1e-05 * Math.Abs(b.X) && Math.Abs(a.Y - b.Y) <= 1e-08 + 1e-05 * Math.Abs(b.Y);

    /// <summary>Rooms and points of one page. Without a legend (and not allowed) only the legend rows come back.
    /// givenPoints: the symbols were already found (a picture: matched as images) - only rooms are read here.
    /// sharedLegend: (shapes, text symbols) of a legend sheet of the set, used when this page has no legend itself</summary>
    private static PageResult ReadPage(PdfDoc doc, int pageNo, List<(TextItem T, PRect R)> lines, bool usedOcr, SymbolMapper mapper,
                                       bool allowNoLegend, List<PointItem>? givenPoints = null, List<List<Pt>>? givenOutlines = null,
                                       (List<(Symbol Sym, string Kind)> Shapes, Dictionary<string, string> Texts)? sharedLegend = null)
    {
        var page = doc[pageNo - 1];
        var texts = lines.Select(l => l.T).ToList();
        var drawings = page.GetDrawings();
        var logs = new List<string>();
        var warns = new List<string>();
        if (usedOcr)                         // word outlines out before the drawing is broken into strokes
        {
            var words = WordPaths(drawings, lines, page.RotationMatrix);
            drawings = drawings.Where((d, i) => !words.Contains(i)).ToList();
        }
        var items = PdfGeometry.Strokes(drawings, page.RotationMatrix);
        if (usedOcr) items = DropWordOutlines(items, lines);
        var titles = texts.Select(t => (Name: Common.FloorName(t.Text), t.X, t.Y)).Where(x => x.Name != null).Select(x => (x.Name!, x.X, x.Y)).ToList();
        if (titles.Select(x => x.Item1).Distinct().Count() == 1) titles = titles.Take(1).ToList();   // the plan title and the title block name the same floor
        var pageFloor = titles.Count == 1 ? titles[0].Item1 : $"Page {pageNo}";
        int? scale = ScaleOf(texts);
        bool hasScale = scale is { } sc0 && sc0 != 0;

        // Step 3 - legend
        var (shapes, textSymbols, legendRect, rows) = Legend(lines, items, mapper);
        (List<(Symbol, string)>, Dictionary<string, string>)? ownLegend = shapes.Count > 0 || textSymbols.Count > 0 ? (shapes, textSymbols) : null;
        bool borrowed = ownLegend == null && sharedLegend != null;
        if (borrowed) (shapes, textSymbols) = (sharedLegend!.Value.Shapes, sharedLegend.Value.Texts);
        if (legendRect is { } lrc)           // legend text is not a floor title ("... WEATHER PROOF")
        {
            titles = titles.Where(tt => !lrc.Contains(tt.Item2, tt.Item3)).ToList();
            if (titles.Select(x => x.Item1).Distinct().Count() == 1) titles = titles.Take(1).ToList();
            pageFloor = titles.Count == 1 ? titles[0].Item1 : $"Page {pageNo}";
        }
        if (givenPoints != null) allowNoLegend = true;          // (the legend was read from the picture itself)
        if (shapes.Count == 0 && textSymbols.Count == 0 && !allowNoLegend)
            return new PageResult { Page = pageNo, Legend = false, Rows = rows };
        var lr = legendRect is { } lrr ? T4(lrr) : ((double, double, double, double)?)null;
        var planItems = items.Where(s => !(lr is { } l && s.Inside(l))).ToList();

        // Step 5 - symbols: shapes matching a legend symbol, then short texts ("AC") matching one
        var sizes = shapes.Select(s => s.Sym.Size).ToList();
        double maxSize = sizes.Count > 0 ? 4 * sizes.Max() : 0.02 * Math.Min(page.Rect.Width, page.Rect.Height);
        double gap = sizes.Count > 0 ? Math.Max(0.3, 0.12 * Py.Median(sizes)) : 0.5;
        double sizeWeight = 0.4;
        if (borrowed)
        {
            // a legend sheet is drawn at its own size, not the plan's: size counts little, and long thin symbols
            // (a strip light) do not set how big a symbol can be (walls would be taken in with the symbols)
            var compact = shapes.Where(s => Math.Min(s.Sym.W, s.Sym.H) >= 0.25 * Math.Max(s.Sym.W, s.Sym.H)).Select(s => s.Sym.Size).ToList();
            // (a legend of text symbols only - "AC", "TV" - gives no shape size: the plan's default stays)
            if (compact.Count > 0 || sizes.Count > 0) maxSize = 2 * (compact.Count > 0 ? compact : sizes).Max();
            sizeWeight = 0.1;
        }
        // strokes in the legend symbols' colours only; all of them when the legend is in black / one colour
        // (not a legend on another sheet: the plans can be drawn in other colours)
        var colours = borrowed ? new HashSet<string>() : shapes.Where(s => s.Sym.Color.Length > 0).Select(s => s.Sym.Color).ToHashSet();
        var dark = colours.Where(c => c.Split(',').All(v => int.Parse(v, Py.Inv) <= 2)).ToHashSet();
        var found = givenPoints != null ? new List<Symbol>() : PdfGeometry.FindSymbols(
            planItems, maxSize, gap, colours.Except(dark).Any() ? colours : null,
            maxLine: sizes.Count > 0 ? 1.5 * Py.Median(sizes) : null, sameColour: borrowed);
        var shortTexts = lines.Select((l, k) => (k, l)).Where(x => x.l.T.Text.Trim().Length <= 4 && !(legendRect is { } lr2 && lr2.Intersects(x.l.R)))
                              .Select(x => (K: x.k, T: x.l.T)).ToList();
        // point types drawn as a filled dot (switches): the area of one dot, per colour
        var dotKinds = new Dictionary<string, (string, double)>();
        foreach (var (sym, kind) in borrowed ? new List<(Symbol, string)>() : shapes)     // (dot sizes on another sheet say nothing)
            if (sym.Filled && sym.FillArea > 0 && sym.Color.Length > 0) dotKinds.TryAdd(sym.Color, (kind, sym.FillArea));
        var points = new List<PointItem>(givenPoints ?? new List<PointItem>());
        var unknown = new List<Symbol>();
        var symbolTexts = new HashSet<int>();      // indexes into lines: texts that are (part of) a symbol
        var legendTexts = textSymbols.Keys.Concat(shapes.Where(s => s.Sym.Text.Length > 0).Select(s => s.Sym.Text)).ToHashSet();
        foreach (var sym in found)
        {
            var (x0, y0, x1, y1) = sym.BBox;
            foreach (var (k, t) in usedOcr ? new List<(int, TextItem)>() : shortTexts)    // OCR'd letters stay part of the shape instead
            {
                // (a room name next to a symbol - "BED" under a fan - is not the symbol's text, unless the legend's
                // symbols carry that text; two letters - "TV" in a TV point - are the symbol's)
                if (Common.Norm(t.Text).Length >= 3 && Common.RoomType(t.Text) != null && !legendTexts.Contains(Common.Norm(t.Text))) continue;
                if (x0 <= t.X && t.X <= x1 && y0 <= t.Y && t.Y <= y1)
                {
                    sym.Text = Common.Norm(t.Text);
                    symbolTexts.Add(k);
                    break;
                }
            }
            var (kindM, _) = PdfGeometry.BestMatch(sym, shapes, sizeWeight);
            if (!string.IsNullOrEmpty(kindM)) points.Add(new PointItem(kindM, sym.X, sym.Y));
            else if (Dots(sym, dotKinds) is { } nd)
            {
                // touching dots (switches stacked on a wall): one point per dot's worth of filled area
                var p = new PointItem(nd.Kind, sym.X, sym.Y);
                for (int q = 0; q < nd.N; q++) points.Add(p);
            }
            else unknown.Add(sym);
        }
        foreach (var (k, t) in shortTexts)
        {
            if (textSymbols.TryGetValue(Common.Norm(t.Text), out var kind) && !string.IsNullOrEmpty(kind) && !symbolTexts.Contains(k))
            {
                points.Add(new PointItem(kind, t.X, t.Y));
                symbolTexts.Add(k);
            }
        }
        var unknownGroups = new List<List<Symbol>>();
        if (shapes.Count == 0 && givenPoints == null)
        {
            unknownGroups = PdfGeometry.GroupSameShapes(unknown.Take(1500).ToList());
            if (unknownGroups.Count > 0)
                warns.Add($"Page {pageNo}: {unknownGroups.Count} repeated symbol shape(s) found but not counted (no " +
                          "legend says what they are): " + string.Join(", ", unknownGroups.Take(8).Select((g, i) => $"shape {(char)(65 + i)} x{g.Count}")));
        }

        // Step 4 - rooms
        var labels = RoomLabels(lines, legendRect, symbolTexts);
        // several drawings on one sheet (floor plans side by side, an elevation, a section, an area plan): each is a
        // floor / section of its own; names in elevations and sections are not rooms
        var views = Views(items, lines, page.Rect, legendRect);
        bool multi = views.Count >= 2 && views.Count(v => v.Titled) >= 2;    // (a title block title alone is not it)
        View? ViewOf(double x, double y) => views.FirstOrDefault(v => v.Box.X0 <= x && x <= v.Box.X1 && v.Box.Y0 <= y && y <= v.Box.Y1);
        if (multi)
        {
            int plans = 0;
            foreach (var v in views)
            {
                bool inside = labels.Any(lb => ReferenceEquals(ViewOf((lb.Rect.X0 + lb.Rect.X1) / 2, (lb.Rect.Y0 + lb.Rect.Y1) / 2), v));
                if (!v.Titled && inside) v.Kind = "plan";
                if (v.Kind == "plan" && !v.Titled)
                {
                    plans++;
                    v.Name = $"Plan {plans} (untitled)";
                }
                if (NotPlan.Contains(v.Kind) || v.Kind == "other") v.Name ??= "Other drawing";
            }
            labels = labels.Where(lb => ViewOf((lb.Rect.X0 + lb.Rect.X1) / 2, (lb.Rect.Y0 + lb.Rect.Y1) / 2) is { } v && v.Kind == "plan").ToList();
            points = points.Where(p => !(ViewOf(p.X, p.Y) is { } v && NotPlan.Contains(v.Kind))).ToList();
            logs.Add($"Page {pageNo}: {views.Count} drawings on the sheet - " + string.Join(", ", views.Select(v =>
                $"{v.Name ?? "None"} ({(v.Kind == "plan" ? "floor plan" : v.Kind + (NotPlan.Contains(v.Kind) ? " - not a floor plan" : ""))})")));
        }
        // no scale written (N.T.S.): from the dimension strings, else a guess from how far apart the room names are
        var dimsScale = hasScale ? null : FeetFromDimensions(lines, planItems);
        double ft = hasScale ? 72 * 12.0 / scale!.Value : dimsScale is { } ds ? ds.Ft : FeetGuess(labels, page);
        if (dimsScale is { } ds2) logs.Add($"Page {pageNo}: no scale written - worked out from {ds2.Agree} dimension line(s)");
        // (no room reaches over these: titles, the legend, and a void open to the floor below - "LOOK BELOW")
        var forbidden = texts.Where(t => Common.FloorName(t.Text) != null || Common.IsLegendTitle(Common.Norm(t.Text)) || TitleText(t.Text)
                                         || OpenBelow.IsMatch(Common.Norm(t.Text))).Select(t => (t.X, t.Y)).ToList();
        if (legendRect is { } lr3) forbidden.Add(((lr3.X0 + lr3.X1) / 2, (lr3.Y0 + lr3.Y1) / 2));
        var areaRange = hasScale || dimsScale != null ? (12.0, 6000.0) : (2.0, 30000.0);
        var finder = new RoomFinder(planItems, page.Rect, ft, areaRange);
        var building = multi ? null : finder.Footprint();       // (several drawings: their own areas did that above)
        if (building is { } bld)       // names outside the building's walls (title block, notes, details) are not rooms
            labels = labels.Where(lb => bld.X0 <= (lb.Rect.X0 + lb.Rect.X1) / 2 && (lb.Rect.X0 + lb.Rect.X1) / 2 <= bld.X1
                                        && bld.Y0 <= (lb.Rect.Y0 + lb.Rect.Y1) / 2 && (lb.Rect.Y0 + lb.Rect.Y1) / 2 <= bld.Y1).ToList();
        var centres = labels.Select(lb => ((lb.Rect.X0 + lb.Rect.X1) / 2, (lb.Rect.Y0 + lb.Rect.Y1) / 2, lb.Rect.Width / 2)).ToList();
        var outlines = finder.Find(centres, forbidden);
        var dimsTexts = lines.Select(l => (l.T, l.R, D: Common.ParseDimensions(l.T.Text))).Where(x => x.D != null)
                             .Select(x => (x.T, x.R, D: x.D!.Value)).ToList();

        // no scale written: work it out from rooms whose size is written ("14'x12'" in a traced outline)
        double? sqft = hasScale || dimsScale != null ? 1 / Math.Pow(ft, 2) : null;
        if (!hasScale && dimsScale == null)
        {
            var ratios = new List<double>();
            for (int i = 0; i < labels.Count; i++)
            {
                var lb = labels[i];
                var poly = outlines[i];
                if (poly == null) continue;
                var inside = dimsTexts.Where(d => poly.Covers(Geo.Box(d.T.X - 0.1, d.T.Y - 0.1, d.T.X + 0.1, d.T.Y + 0.1))).Select(d => d.D).ToList();
                // (or the size written right under the name - as a room's area is matched below)
                double cx = (lb.Rect.X0 + lb.Rect.X1) / 2;
                var below = dimsTexts.Where(d => Math.Abs(d.T.X - cx) < 6 * lb.H && 0 <= d.R.Y0 - lb.Rect.Y0 && d.R.Y0 - lb.Rect.Y0 < 3 * lb.H + lb.Rect.Height)
                                     .Select(d => d.D).ToList();
                if (inside.Count > 0 || below.Count > 0) ratios.Add((inside.Count > 0 ? inside : below)[0].Sqft / poly.Area);
            }
            double? better = ratios.Count > 0 ? 1 / Math.Pow(Py.Median(ratios), 0.5) : null;
            // a room or two far off the guess: their outlines are likely pieces of the rooms (furniture) - not trusted
            if (ratios.Count > 0 && ratios.Count < 3 && !(0.6 < better!.Value / ft && better.Value / ft < 1.6)) ratios.Clear();
            if (ratios.Count > 0)
            {
                sqft = Py.Median(ratios);
                // the guess was off: trace again with the right door width (when enough rooms agree on it)
                if (Math.Abs(better!.Value / ft - 1) > 0.1 && ratios.Count >= 3)
                {
                    ft = better.Value;
                    outlines = new RoomFinder(planItems, page.Rect, ft, areaRange).Find(centres, forbidden);
                }
                logs.Add($"Page {pageNo}: no scale written - worked out from {ratios.Count} room(s) with dimensions");
            }
        }

        // rooms not traced: a room outline found in the picture itself (RasterReader) around the name
        if (givenOutlines is { Count: > 0 })
        {
            var shapes2 = givenOutlines.Where(o => o.Count >= 3).Select(o => Geo.Buffer(Geo.Polygon(o), 0)).Where(p => !p.IsEmpty).ToList();
            for (int i = 0; i < labels.Count; i++)
            {
                var lb = labels[i];
                var c = Geo.Point((lb.Rect.X0 + lb.Rect.X1) / 2, (lb.Rect.Y0 + lb.Rect.Y1) / 2);
                var inside = shapes2.Where(p => p.Contains(c)).ToList();
                if (inside.Count == 0) continue;
                var best = Py.MinBy(inside, p => p.Area);
                // (also when the traced outline is a small part of it: furniture around the name, not the room)
                if (outlines[i] == null || outlines[i]!.Area < 0.4 * best.Area) outlines[i] = best;
            }
        }

        // rooms not traced: the smallest drawn rectangle around the name
        var boxes = new List<PRect>();
        foreach (var s in planItems)
        {
            var (x0, y0, x1, y1) = s.BBox;
            if (x1 - x0 > 40 && y1 - y0 > 40 && s.Pts.Length == 5 && AllClose(s.Pts[0], s.Pts[^1]))
            {
                if (building is { } b2 && (x0 < b2.X0 - ft || y0 < b2.Y0 - ft || x1 > b2.X1 + ft || y1 > b2.Y1 + ft))
                    continue;                  // bigger than the building (plot limit, sheet frame)
                boxes.Add(PRect.From(s.BBox));
            }
        }
        var rooms = new List<Room>();
        int traced = 0;
        // names that are not in any room, away from the traced rooms (title block, notes): not rooms
        PRect? planArea = null;
        var polys = outlines.Where(o => o != null).Select(o => o!).ToList();
        if (polys.Count >= 3)
        {
            double x0 = polys.Min(o => o.EnvelopeInternal.MinX), y0 = polys.Min(o => o.EnvelopeInternal.MinY);
            double x1 = polys.Max(o => o.EnvelopeInternal.MaxX), y1 = polys.Max(o => o.EnvelopeInternal.MaxY);
            double pad = 0.05 * Math.Max(x1 - x0, y1 - y0);
            planArea = new PRect(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
        }
        for (int i = 0; i < labels.Count; i++)
        {
            var (name, kind, rect, h) = (labels[i].Name, labels[i].Kind, labels[i].Rect, labels[i].H);
            var poly = outlines[i];
            double cx = (rect.X0 + rect.X1) / 2, cy = (rect.Y0 + rect.Y1) / 2;
            if (poly == null && planArea is { } pa && !pa.Contains(cx, cy)) continue;
            if (poly != null) traced++;
            else
            {
                // (a room is far bigger than its name: a rectangle hardly wider than the name is furniture - a table)
                var around = boxes.Where(b => b.Contains(cx, cy) && b.Width * b.Height < page.Rect.Width * page.Rect.Height * 0.5
                                              && b.Width >= 2 * rect.Width && b.Height >= 3 * rect.Height).ToList();
                if (around.Count > 0)
                {
                    var b = Py.MinBy(around, x => x.Width * x.Height);
                    poly = Geo.Box(b.X0, b.Y0, b.X1, b.Y1);
                }
            }
            (string Text, double Sqft)? dims = null;
            if (poly != null)
            {
                var inside = dimsTexts.Where(d => poly.Covers(Geo.Box(d.T.X - 0.1, d.T.Y - 0.1, d.T.X + 0.1, d.T.Y + 0.1)))
                                      .Select(d => (Dist: Math.Abs(d.T.X - cx) + Math.Abs(d.T.Y - cy), d.D)).ToList();
                if (inside.Count > 0)
                    dims = inside.OrderBy(x => x.Dist).ThenBy(x => x.D.Text, StringComparer.Ordinal).ThenBy(x => x.D.Sqft).First().D;
            }
            dims ??= dimsTexts.Where(d => Math.Abs(d.T.X - cx) < 6 * h && 0 <= d.R.Y0 - rect.Y0 && d.R.Y0 - rect.Y0 < 3 * h + rect.Height)
                              .Select(d => ((string, double)?)d.D).FirstOrDefault();
            double? area = dims is { } dd ? dd.Sqft : (poly != null && sqft is { } sq && sq != 0 ? Py.Round(poly.Area * sq, 1) : null);
            rooms.Add(new Room(name, poly, cx, cy, kind, dims?.Text, area));
        }
        // spaces found in a picture with no name read ("ROOM n"): one smaller than any real room is a gap in the
        // furniture or the lettering - left out; the rest numbered again
        if (rooms.Any(r => RoomN.IsMatch(r.Name)))
        {
            rooms = rooms.Where(r => !(RoomN.IsMatch(r.Name) && r.AreaSqft != null && r.AreaSqft < 25)).ToList();
            int k = 0;
            foreach (var r in rooms)
                if (RoomN.IsMatch(r.Name))
                {
                    k++;
                    r.Name = $"ROOM {k}";
                }
        }
        // two labels in the same rectangle: keep the room outline only for the first
        var seen = new HashSet<(long, long, long, long)>();
        foreach (var room in rooms)
        {
            if (room.Poly == null) continue;
            var e = room.Poly.EnvelopeInternal;
            var key = ((long)Py.Round(e.MinX), (long)Py.Round(e.MinY), (long)Py.Round(e.MaxX), (long)Py.Round(e.MaxY));
            if (seen.Contains(key))
            {
                room.Poly = null;
                if (room.Dimensions == null) room.AreaSqft = null;     // (its area was that outline's: not counted twice)
            }
            seen.Add(key);
        }
        // a service room (bath, dress, store ...) lying inside another room's outline: that outline took it in through
        // an open doorway or a wall not drawn - it is cut out, so no floor is counted twice. (Not a room of the same
        // kind - a sofa's box named like the room - nor one only partly inside, whose own outline may be the wrong one.)
        var tracedRooms = Py.Sorted(rooms.Where(r => r.Poly != null), r => r.Poly!.Area);
        for (int i = 0; i < tracedRooms.Count; i++)
        {
            var big = tracedRooms[i];
            var cut = tracedRooms.Take(i).Where(s => s.Kind != null && ServiceRooms.Contains(s.Kind) && s.Kind != big.Kind
                                                     && s.Poly!.Intersects(big.Poly!) && Geo.Intersection(s.Poly, big.Poly!).Area > 0.8 * s.Poly.Area)
                                   .Select(s => s.Poly!).ToList();
            if (cut.Count == 0) continue;
            var rest = Geo.Difference(big.Poly!, Geo.UnionAll(cut));
            var parts = Geo.Parts(rest).Where(g => g.Area > 0).ToList();
            if (parts.Count == 0) continue;
            var label = Geo.Point(big.X, big.Y);
            big.Poly = parts.Select((g, idx) => (g, idx)).OrderBy(x => Geo.Buffer(x.g, 1).Contains(label) ? 0 : 1).ThenBy(x => -x.g.Area).ThenBy(x => x.idx).First().g;
            if (big.Dimensions == null && sqft is { } sq2 && sq2 != 0) big.AreaSqft = Py.Round(big.Poly.Area * sq2, 1);
        }
        // an area far beyond what such a room can be: the outline ran out through an opening - no area rather than
        // a wrong one; and far below it: a piece of furniture or a stair tread traced instead of the room
        foreach (var room in rooms)
        {
            double? most = RoomMaxSqft.TryGetValue(room.Kind ?? "", out var mo) ? mo : null;
            double? least = RoomMinSqft.TryGetValue(room.Kind ?? "", out var le) ? le : null;
            if (room.AreaSqft is null or 0) continue;
            if (room.Dimensions != null)     // a written size: only an impossible one - a note's "2 x 4" taken as it
            {
                if (least is { } l1 && l1 != 0 && room.AreaSqft < l1)
                {
                    warns.Add($"Page {pageNo}: {room.Name} - the size found next to it ({room.Dimensions}) is too " +
                              $"small for a {room.Kind}; no area given - check it on the drawing.");
                    room.AreaSqft = null;
                    room.Dimensions = null;
                }
                continue;
            }
            string what;
            if (most is { } m1 && m1 != 0 && room.AreaSqft > m1) what = "too big";
            else if (least is { } l2 && l2 != 0 && room.AreaSqft < l2) what = "too small";
            else continue;
            warns.Add($"Page {pageNo}: {room.Name} - the outline traced ({Py.Fixed(room.AreaSqft.Value, 0)} sq ft) is {what} " +
                      $"for a {room.Kind}; no area given - check it on the drawing.");
            room.AreaSqft = null;
            room.Poly = null;
        }

        // an area plan (the covered area split into parts "A 6'-6" x 7'-0"" ...): each size written in it is an area
        if (multi)
        {
            foreach (var v in views.Where(v => v.Kind == "area").ToList())
            {
                var (x0, y0, x1, y1) = (v.Box.X0, v.Box.Y0, v.Box.X1, v.Box.Y1);
                int k = 0;
                foreach (var (t, r, d) in dimsTexts)
                {
                    double cx = (r.X0 + r.X1) / 2, cy = (r.Y0 + r.Y1) / 2;
                    if (!(x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1)) continue;
                    k++;
                    var tag = lines.Where(l => Py.FullMatch(@"[A-Z]\d?", l.T.Text.Trim()) != null && 0 < r.Y0 - l.R.Y1 && r.Y0 - l.R.Y1 < 4 * r.Height
                                               && Math.Abs((l.R.X0 + l.R.X1) / 2 - cx) < r.Width).Select(l => l.T.Text.Trim()).FirstOrDefault();
                    rooms.Add(new Room($"Area {(string.IsNullOrEmpty(tag) ? k.ToString(Py.Inv) : tag)}", null, cx, cy, null, d.Text, d.Sqft));
                }
            }
        }

        var counts = points.GroupBy(p => p.Kind).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Key} {g.Count()}").ToList();
        logs.Add($"Page {pageNo} ({pageFloor}): legend {shapes.Count + textSymbols.Count} symbol(s), {rooms.Count} room(s) " +
                 $"({traced} traced from walls), {found.Count} symbol shape(s) checked; " + (counts.Count > 0 ? string.Join(", ", counts) : "no points"));
        var sheet = new Sheet(pageFloor)
        {
            Rooms = rooms, Points = points,
            FloorTitles = multi ? new() : (titles.Count > 1 ? titles : new()),
            ConfidencePenalty = 0.1, Snap = 2.0 * ft,
            Views = views.Where(v => multi && (v.Kind == "plan" || v.Kind == "area")).Select(v => (v.Name ?? "", v.Box)).ToList(),
            Debug = new SheetDebug
            {
                Page = pageNo - 1, Legend = legendRect, Unknown = unknownGroups,
                Views = multi ? views.Select(v => new ViewInfo { Page = pageNo, Name = v.Name, Kind = v.Kind }).ToList() : new(),
            },
        };
        return new PageResult
        {
            Page = pageNo, Legend = shapes.Count > 0 || textSymbols.Count > 0 || givenPoints != null, Rows = rows, Sheet = sheet,
            Log = logs, Warnings = warns, OwnLegend = ownLegend,
            LegendShare = ownLegend != null && legendRect is { } lr4 ? lr4.Area / page.Rect.Area : 0.0,
        };
    }

    /// <summary>pages read in parallel (the work is the reader's own; MuPDF calls are taken one at a time)</summary>
    private static List<PageResult> ReadPages(PdfDoc doc, List<(int Page, List<(TextItem T, PRect R)> Lines, bool Ocr)> jobs, SymbolMapper mapper, bool allow,
                                              Dictionary<int, List<PointItem>>? pointsOf = null, Dictionary<int, List<List<Pt>>>? outlinesOf = null,
                                              (List<(Symbol Sym, string Kind)>, Dictionary<string, string>)? shared = null)
    {
        int workers = Math.Min(Math.Max(1, Environment.ProcessorCount - 1),
                               int.TryParse(Environment.GetEnvironmentVariable("D2Q_WORKERS"), out var w) && w > 0 ? w : 8);
        var results = new PageResult[jobs.Count];
        Parallel.For(0, jobs.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, workers) }, i =>
        {
            var (n, ln, o) = jobs[i];
            results[i] = ReadPage(doc, n, ln, o, mapper, allow,
                                  pointsOf != null && pointsOf.TryGetValue(n, out var p) ? p : null,
                                  outlinesOf != null && outlinesOf.TryGetValue(n, out var ol) ? ol : null, shared);
        });
        return results.ToList();
    }

    // words in an electrical sheet's title ("GROUND FLOOR LIGHTING PLAN", "POWER & AC LAYOUT", "ELECTRICAL PLAN")
    private static readonly Regex ElectricalSheet = new(@"\b(ELECTRICAL|LIGHTING|POWER|SOCKETS?|SWITCH(ES|ING)?|DATA|CCTV|WIRING|CONDUITS?)\b", Py.RX);

    /// <summary>pages: 1-based page numbers to read (all when null).
    /// ocrLines: page number -> text lines already read by OCR (a drawing traced from a picture: RasterReader)
    /// pointsOf: page number -> the points already found there (a picture's symbols, matched as images)</summary>
    public static (Dictionary<string, string> Project, List<Sheet> Sheets, List<LegendRow> Legend) Read(
        byte[] data, SymbolMapper mapper, List<string> log, List<string> warnings, IList<int>? pages = null,
        Dictionary<int, List<(TextItem T, PRect R)>>? ocrLines = null, Dictionary<int, List<PointItem>>? pointsOf = null,
        Dictionary<int, List<List<Pt>>>? outlinesOf = null)
    {
        using var doc = new PdfDoc(data);
        var docHash = Ocr.Sha1(data);
        var project = new Dictionary<string, string>();
        var legendAll = new List<KeyValuePair<string, LegendRow>>();
        var ocrPages = new List<int>();
        int pageCount = doc.PageCount;
        if (ocrLines == null) log.Add($"Vector PDF, {pageCount} page(s) - each page is read as a floor");
        log.Add("Step 1/2 - title block and floor titles are read from the page text");
        // 1. the text of every page (OCR for printed pages - it uses all CPU cores itself, so one page at a time)
        var textsOf = new Dictionary<int, (List<(TextItem T, PRect R)> Lines, bool Ocr)>();
        var order = new List<int>();
        foreach (int pageNo in Enumerable.Range(1, pageCount).Where(n => pages == null || pages.Count == 0 || pages.Contains(n)))
        {
            var page = doc[pageNo - 1];
            List<(TextItem T, PRect R)> lines;
            bool usedOcr;
            if (ocrLines != null)
            {
                lines = ocrLines.TryGetValue(pageNo, out var l) ? l : new();
                usedOcr = true;
            }
            else
            {
                lines = Lines(page);
                usedOcr = false;
            }
            if (lines.Count == 0 && !usedOcr && Ocr.Available())
            {
                Ocr.Progress?.Invoke(pageNo, pageCount);
                lines = Ocr.PageLines(page, docHash);
                usedOcr = lines.Count > 0;
                if (usedOcr) ocrPages.Add(pageNo);
            }
            textsOf[pageNo] = (lines, usedOcr);
            order.Add(pageNo);
            foreach (var (k, v) in Common.TitleBlock(lines.Select(x => x.T).ToList())) project.TryAdd(k, v);
        }
        List<(int, List<(TextItem T, PRect R)>, bool)> Jobs(IEnumerable<int> ns) => ns.Select(n => (n, textsOf[n].Lines, textsOf[n].Ocr)).ToList();
        // 2. rooms and points, pages in parallel. Pages without a legend are read for them only when no page of the
        // drawing has a legend (a drawing set has its legends on the layout sheets; cover, registers, notes, details
        // have none).
        var withHeading = order.Where(n => textsOf[n].Lines.Any(l => LegendWord(l.T.Text))).ToList();
        if (pointsOf != null)                 // a picture: every page with points found is read
            withHeading = withHeading.Union(pointsOf.Where(kv => kv.Value.Count > 0).Select(kv => kv.Key)).Distinct().OrderBy(n => n).ToList();
        var results = ReadPages(doc, Jobs(withHeading), mapper, false, pointsOf, outlinesOf);
        bool legendSeen = results.Any(r => r.Legend);
        // a legend sheet whose legend words are drawn as lines (CAD fonts exported as shapes), only the title block
        // being text: its words are read from the image (OCR) as well
        var lettered = new List<int>();
        if (!legendSeen && ocrLines == null && Ocr.Available())
        {
            foreach (var r in results)
            {
                int n = r.Page;
                var (lines, used) = textsOf[n];
                if (used || !lines.Any(l => Common.IsLegendTitle(Common.Norm(l.T.Text)) || Py.Search(@"\bLEGENDS?\b", Common.Norm(l.T.Text)))) continue;
                Ocr.Progress?.Invoke(n, pageCount);
                var extra = Ocr.PageLines(doc[n - 1], docHash)
                               .Where(x => !lines.Any(l => (x.R & l.R).Area > 0.5 * x.R.Area)).ToList();
                if (extra.Count > 0)
                {
                    textsOf[n] = (lines.Concat(extra).ToList(), true);
                    lettered.Add(n);
                }
            }
            if (lettered.Count > 0)
            {
                var again = ReadPages(doc, Jobs(lettered), mapper, false, pointsOf, outlinesOf).ToDictionary(x => x.Page);
                results = results.Select(r => again.TryGetValue(r.Page, out var a) ? a : r).ToList();
                legendSeen = results.Any(r => r.Legend);
            }
        }
        // the legend on a sheet of its own (the legend fills a good part of it, hardly any points - fewer than the
        // legend has symbols): the plan sheets without a legend use it. The legend sheet is not a floor (what it found
        // outside the legend are strays).
        var legendSheets = results.Where(r => r.OwnLegend != null && r.Sheet != null && r.LegendShare >= 0.25
                                              && r.Sheet.Points.Count < r.OwnLegend.Value.Shapes.Count + r.OwnLegend.Value.Texts.Count).ToList();
        if (legendSheets.Count > 0 && legendSheets.Count < textsOf.Count)
        {
            foreach (var r in legendSheets) r.Sheet = null;
            var sharedTexts = new Dictionary<string, string>();
            foreach (var r in legendSheets)
                foreach (var (k, v) in r.OwnLegend!.Value.Texts) sharedTexts[k] = v;
            var shared = (legendSheets.SelectMany(r => r.OwnLegend!.Value.Shapes).ToList(), sharedTexts);
            var withLegend = results.Where(r => r.Legend).Select(r => r.Page).ToHashSet();
            var rest = textsOf.Keys.Where(n => !withLegend.Contains(n)).OrderBy(n => n).ToList();
            // a full working set (architecture, plumbing, structure ...): only the electrical layout sheets
            var electrical = rest.Where(n => textsOf[n].Lines.Any(l => Py.Split(l.T.Text).Length <= 8 && ElectricalSheet.IsMatch(Common.Norm(l.T.Text)))).ToList();
            if (electrical.Count > 0) rest = electrical;
            if (rest.Count > 0)
            {
                var more = ReadPages(doc, Jobs(rest), mapper, false, pointsOf, outlinesOf, shared);
                // (a sheet with none of the legend's symbols - cover, notes, details - is not a plan; nor one with no
                // floor named and hardly any rooms: a schematic, a panel detail)
                more = more.Where(r => r.Sheet != null && r.Sheet.Points.Count > 0
                                       && (!r.Sheet.DefaultFloor.StartsWith("Page ") || r.Sheet.Rooms.Count >= 5)).ToList();
                results.AddRange(more);
                var used = PagesText(more.Select(r => r.Page).OrderBy(n => n).ToList());
                log.Add($"Legend on page(s) {PagesText(legendSheets.Select(r => r.Page).ToList())} used for page(s) " +
                        (used.Length > 0 ? used : "none"));
            }
        }
        var legendPages = results.Where(r => r.Legend).Select(r => r.Page).ToHashSet();
        var noLegend = textsOf.Keys.Where(n => !legendPages.Contains(n)).OrderBy(n => n).ToList();
        if (!legendSeen && noLegend.Count > 0)
            results = ReadPages(doc, Jobs(textsOf.Keys.OrderBy(n => n)), mapper, true, pointsOf, outlinesOf);
        var sheets = new List<Sheet>();
        foreach (var r in Py.Sorted(results, r => r.Page))
        {
            foreach (var kv in r.Rows)
                if (!legendAll.Any(x => x.Key == kv.Key)) legendAll.Add(kv);
            if (r.Sheet != null)
            {
                sheets.Add(r.Sheet);
                log.AddRange(r.Log);
                warnings.AddRange(r.Warnings);
            }
        }
        if (ocrPages.Count > 0)
            warnings.Add($"Page(s) {PagesText(ocrPages)} had no text (printed to PDF) - their words were read from the " +
                         "image (OCR); check room names and counts.");
        if (lettered.Count > 0)
            warnings.Add($"Page(s) {PagesText(lettered)}: the legend's words are drawn as lines, not text - they were read " +
                         "from the image (OCR); check the legend.");
        if (noLegend.Count > 0 && legendSeen)
            warnings.Add($"Page(s) {PagesText(noLegend)}: no legend found - only the title block was read on these pages.");
        else if (noLegend.Count > 0)
            warnings.Add($"Page(s) {PagesText(noLegend)}: no legend found - symbols on these pages are not counted.");
        log.Add("Step 3 - legend: " + (legendAll.Count > 0 ? string.Join(", ", legendAll.Select(l => $"{l.Value.Description}={l.Value.Type}")) : "none"));
        return (project, sheets, legendAll.Select(kv => kv.Value).ToList());
    }

    /// <summary>1, 2, 3, 5 -> "1-3, 5"</summary>
    internal static string PagesText(IList<int> numbers)
    {
        var out_ = new List<string>();
        int? start = null;
        for (int i = 0; i < numbers.Count; i++)
        {
            int n = numbers[i];
            start ??= n;
            if (i + 1 == numbers.Count || numbers[i + 1] != n + 1)
            {
                out_.Add(start == n ? n.ToString(Py.Inv) : $"{start}-{n}");
                start = null;
            }
        }
        return string.Join(", ", out_);
    }
}
