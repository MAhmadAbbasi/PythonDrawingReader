using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Drawing2Quote.Reader;

/// <summary>What was read from a drawing - the same JSON the Python service's /extract returns</summary>
public sealed class Extraction
{
    [JsonPropertyName("fileName")] public string FileName { get; set; } = "";
    [JsonPropertyName("fileType")] public string FileType { get; set; } = "";
    /// <summary>title block: projectName, client, drawingNumber, scale, date, architect, plotSize</summary>
    [JsonPropertyName("project")] public Dictionary<string, string> Project { get; set; } = new();
    [JsonPropertyName("floors")] public List<Floor> Floors { get; set; } = new();
    [JsonPropertyName("legend")] public List<LegendRow> Legend { get; set; } = new();
    [JsonPropertyName("totals")] public Totals Totals { get; set; } = new();
    [JsonPropertyName("sections")] public List<Section> Sections { get; set; } = new();
    /// <summary>false: an architectural layout (no electrical legend or points)</summary>
    [JsonPropertyName("electrical")] public bool Electrical { get; set; }
    [JsonPropertyName("warnings")] public List<string> Warnings { get; set; } = new();
    [JsonPropertyName("log")] public List<string> Log { get; set; } = new();
    [JsonPropertyName("pointTypes")] public List<string> PointTypes { get; set; } = new();

    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public string ToJson(bool indented = true) =>
        JsonSerializer.Serialize(this, indented ? JsonOptions : new JsonSerializerOptions(JsonOptions) { WriteIndented = false });
}

public sealed class Floor
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("areas")] public List<Area> Areas { get; set; } = new();
    [JsonPropertyName("areaSqft")] public double AreaSqft { get; set; }
}

public sealed class Area
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("roomType")] public string? RoomType { get; set; }
    [JsonPropertyName("areaSqft")] public double? AreaSqft { get; set; }
    [JsonPropertyName("dimensions")] public string? Dimensions { get; set; }
    /// <summary>point type -> count, sorted by type</summary>
    [JsonPropertyName("items")] public Dictionary<string, int> Items { get; set; } = new();
    [JsonPropertyName("confidence")] public double Confidence { get; set; }

    // while the sheets are put together
    [JsonIgnore] internal int SheetNo;
    [JsonIgnore] internal (double X, double Y) Pos;
    [JsonIgnore] internal HashSet<int>? Sheets;
}

public sealed class Totals
{
    [JsonPropertyName("areaSqft")] public double AreaSqft { get; set; }
    [JsonPropertyName("items")] public Dictionary<string, int> Items { get; set; } = new();
    [JsonPropertyName("points")] public int Points { get; set; }
}

public sealed class LegendRow
{
    [JsonPropertyName("symbol")] public string Symbol { get; set; } = "";
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("type")] public string Type { get; set; } = "";
}

/// <summary>A drawing found on a sheet: floor plan, area plan, elevation, section ...</summary>
public sealed class Section
{
    [JsonPropertyName("page")] public int Page { get; set; }
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("kind")] public string Kind { get; set; } = "";
    [JsonPropertyName("rooms")] public int Rooms { get; set; }
    [JsonPropertyName("areaSqft")] public double? AreaSqft { get; set; }
}

/// <summary>The file can't be read; the message tells the user why and what to upload instead</summary>
public sealed class RejectedFileException : Exception
{
    public RejectedFileException(string message) : base(message) { }
}
