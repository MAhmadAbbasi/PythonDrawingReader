using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using YamlDotNet.RepresentationModel;

namespace Drawing2Quote.Service.Rules;

/// <summary>Python's dict / value rules on System.Text.Json nodes: the rules work on the JSON the POS sends (the
/// drawing's extraction, the company's catalogue) as the Python service works on its dicts.</summary>
internal static class J
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    public static JsonNode? Get(JsonNode? o, string key) => o is JsonObject obj && obj.TryGetPropertyValue(key, out var v) ? v : null;

    public static bool Has(JsonNode? o, string key) => o is JsonObject obj && obj.ContainsKey(key);

    public static IEnumerable<JsonNode?> Items(JsonNode? o) => o as JsonArray ?? Enumerable.Empty<JsonNode?>();

    public static JsonObject Obj(JsonNode? o) => o as JsonObject ?? new JsonObject();

    public static string? Str(JsonNode? v) => v == null ? null : v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : PyStr(v);

    /// <summary>str(value) as Python writes it (ints without ".0", floats with their shortest form)</summary>
    public static string PyStr(JsonNode? v)
    {
        if (v == null) return "None";
        switch (v.GetValueKind())
        {
            case JsonValueKind.String: return v.GetValue<string>();
            case JsonValueKind.True: return "True";
            case JsonValueKind.False: return "False";
            case JsonValueKind.Number:
                var raw = v.ToJsonString();
                if (long.TryParse(raw, NumberStyles.Integer, Inv, out var l)) return l.ToString(Inv);
                return Drawing2Quote.Reader.Py.Repr(double.Parse(raw, Inv));
            default: return v.ToJsonString();
        }
    }

    /// <summary>a number (int, float, or numeric text); null when it isn't one</summary>
    public static double? Num(JsonNode? v)
    {
        if (v == null) return null;
        switch (v.GetValueKind())
        {
            case JsonValueKind.Number: return double.Parse(v.ToJsonString(), Inv);
            case JsonValueKind.True: return 1;
            case JsonValueKind.False: return 0;
            case JsonValueKind.String:
                return double.TryParse(v.GetValue<string>().Trim(), NumberStyles.Float, Inv, out var d) ? d : null;
            default: return null;
        }
    }

    public static long Int(JsonNode? v) => (long)(Num(v) ?? 0);

    /// <summary>a number as Python would write it: a whole number from an int stays an int</summary>
    public static JsonNode N(double v) => Math.Abs(v) < 9e15 && v == Math.Floor(v) ? JsonValue.Create((long)v) : JsonValue.Create(v);

    public static JsonNode? Clone(JsonNode? v) => v?.DeepClone();

    /// <summary>Python truthiness</summary>
    public static bool Truthy(JsonNode? v) => v switch
    {
        null => false,
        JsonObject o => o.Count > 0,
        JsonArray a => a.Count > 0,
        _ => v.GetValueKind() switch
        {
            JsonValueKind.String => v.GetValue<string>().Length > 0,
            JsonValueKind.Number => Num(v) != 0,
            JsonValueKind.True => true,
            _ => false,
        },
    };

    public static List<string> Strings(JsonNode? v) => Items(v).Select(x => Str(x) ?? "").ToList();

    /// <summary>a YAML file as JSON (scalars: ints, floats, booleans, null as yaml.safe_load gives them)</summary>
    public static JsonNode? FromYaml(string text)
    {
        var yaml = new YamlStream();
        using (var r = new StringReader(text)) yaml.Load(r);
        return yaml.Documents.Count == 0 ? null : Convert(yaml.Documents[0].RootNode);
    }

    private static JsonNode? Convert(YamlNode n)
    {
        switch (n)
        {
            case YamlMappingNode m:
                var o = new JsonObject();
                foreach (var (k, v) in m.Children) o[((YamlScalarNode)k).Value!] = Convert(v);
                return o;
            case YamlSequenceNode s:
                var a = new JsonArray();
                foreach (var c in s.Children) a.Add(Convert(c));
                return a;
            case YamlScalarNode sc:
                var t = sc.Value ?? "";
                if (sc.Style is YamlDotNet.Core.ScalarStyle.SingleQuoted or YamlDotNet.Core.ScalarStyle.DoubleQuoted) return JsonValue.Create(t);
                if (t is "" or "~" or "null" or "Null" or "NULL") return null;
                if (t is "true" or "True" or "TRUE" or "yes" or "Yes" or "on" or "On") return JsonValue.Create(true);
                if (t is "false" or "False" or "FALSE" or "no" or "No" or "off" or "Off") return JsonValue.Create(false);
                if (long.TryParse(t, NumberStyles.AllowLeadingSign, Inv, out var l)) return JsonValue.Create(l);
                if (double.TryParse(t, NumberStyles.Float, Inv, out var d) && t.Any(char.IsDigit)) return JsonValue.Create(d);
                return JsonValue.Create(t);
            default: return null;
        }
    }
}
