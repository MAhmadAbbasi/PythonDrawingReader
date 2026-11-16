using System.Text.Json.Nodes;
using Drawing2Quote.Service.Rules;
using Xunit;

namespace Drawing2Quote.Reader.Tests;

/// <summary>The solution rules (/suggest, /rules) give the Python service's answers: 51 requests (3 drawings, 2
/// catalogues, both plans, Premium confirmations, size override, another company's areas, prices, installation)
/// answered by the Python rules (branch main) in data/python_suggest_cases.json.</summary>
public class SolutionTests
{
    private static readonly JsonObject Cases = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "data", "python_suggest_cases.json")))!;

    public static IEnumerable<object[]> Names() => ((JsonArray)Cases["cases"]!).Select(c => new object[] { (string)c!["name"]! });

    [Theory]
    [MemberData(nameof(Names))]
    public void SameAnswerAsPythonRules(string name)
    {
        var c = ((JsonArray)Cases["cases"]!).First(x => (string)x!["name"]! == name)!;
        var req = (JsonObject)c["request"]!;
        JsonNode got;
        try
        {
            var result = Solution.Suggest((JsonObject)req["extraction"]!, (JsonObject)req["catalog"]!, (string?)req["plan"], req["choices"] as JsonObject,
                                          J.Num(req["sizeId"]) is { } s ? (long)s : null, otherBrand: req["otherBrand"] as JsonObject,
                                          installation: J.Num(req["installation"]));
            if (J.Truthy(req["client"]) && result["breakdown"] is JsonObject b) b["client"] = J.Str(req["client"]);
            got = result;
        }
        catch (SolutionException e)
        {
            got = new JsonObject { ["error"] = e.Message };
        }
        var diffs = new List<string>();
        Compare(c["response"], got, "", diffs);
        Assert.True(diffs.Count == 0, string.Join("\n", diffs.Take(20)));
    }

    [Fact]
    public void SameRulesAsPython()
    {
        var diffs = new List<string>();
        Compare(Cases["rules"], Solution.RuleList(), "", diffs);
        Assert.True(diffs.Count == 0, string.Join("\n", diffs));
    }

    internal static void Compare(JsonNode? a, JsonNode? b, string path, List<string> diffs)
    {
        switch (a)
        {
            case JsonObject oa when b is JsonObject ob:
                if (!oa.Select(kv => kv.Key).SequenceEqual(ob.Select(kv => kv.Key)))
                    diffs.Add($"{path}: keys [{string.Join(",", oa.Select(kv => kv.Key))}] != [{string.Join(",", ob.Select(kv => kv.Key))}]");
                foreach (var (k, v) in oa) if (ob.ContainsKey(k)) Compare(v, ob[k], $"{path}.{k}", diffs);
                break;
            case JsonArray la when b is JsonArray lb:
                if (la.Count != lb.Count) diffs.Add($"{path}: {la.Count} items != {lb.Count}");
                for (int i = 0; i < Math.Min(la.Count, lb.Count); i++) Compare(la[i], lb[i], $"{path}[{i}]", diffs);
                break;
            case JsonValue va when b is JsonValue vb && va.GetValueKind() == System.Text.Json.JsonValueKind.Number
                                   && vb.GetValueKind() == System.Text.Json.JsonValueKind.Number:
                double x = J.Num(va)!.Value, y = J.Num(vb)!.Value;
                if (Math.Abs(x - y) > 1e-9 * Math.Max(1, Math.Abs(x))) diffs.Add($"{path}: {x} != {y}");
                break;
            default:
                if (a?.ToJsonString() != b?.ToJsonString()) diffs.Add($"{path}: {a?.ToJsonString()} != {b?.ToJsonString()}");
                break;
        }
    }
}
