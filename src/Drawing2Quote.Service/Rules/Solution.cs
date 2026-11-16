using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Drawing2Quote.Reader;

namespace Drawing2Quote.Service.Rules;

/// <summary>The rules were refused (a plan that isn't Standard or Premium): the message is for the user</summary>
public sealed class SolutionException : Exception
{
    public SolutionException(string message) : base(message) { }
}

/// <summary>
/// The suggested solution for a reviewed drawing: the building size's package (from the POS), with the items the
/// drawing's points decide replaced by what the drawing needs, filtered by the plan:
///   Standard: Standard-only and "both" products - never asks
///   Premium:  Premium-only products and "both" products; a "both" product that has a Premium-only product of the
///             same group is asked about first (use it / skip it / use the Premium-only product instead)
/// The same rules and answers as the Python service's app/rules (suggest.py, building_size.py, breakdown.py).
/// </summary>
public static class Solution
{
    public static readonly string[] Plans = { "standard", "premium" };
    public const string MatterGateway = "Matter gateway";   // the rule's name; its product is the company's (ruleItems)

    public static JsonObject LoadRules(string? path = null)
    {
        var file = path ?? Path.Combine(Common.ConfigDir, "solution_rules.yaml");
        return J.FromYaml(File.ReadAllText(file)) as JsonObject ?? new JsonObject();
    }

    private static double Ceil(double v) => Math.Ceiling(v);

    /// <summary>A package quantity (decimal in the POS, e.g. 12.5 m of rail): a whole number stays an int</summary>
    private static double Qty(JsonNode? value)
    {
        if (!J.Truthy(value)) return 0;
        var q = J.Num(value);
        if (q == null) return 0;
        return q.Value == Math.Floor(q.Value) ? Math.Truncate(q.Value) : Py.Round(q.Value, 3);
    }

    /// <summary>The drawing rules a company maps its products to (Solution Packages > Drawing rules): name, the gangs a
    /// rule has ("" = one list), what it counts - and the Matter gateway (one per area of another company's automation)</summary>
    public static JsonArray RuleList(JsonObject? rules = null)
    {
        rules ??= LoadRules();
        var out_ = new JsonArray();
        foreach (var r in J.Items(J.Get(rules, "rules")))
        {
            var gangs = new JsonArray();
            if (J.Str(J.Get(r, "rule")) == "gangs")
                for (long g = J.Has(r, "max_gang") ? J.Int(J.Get(r, "max_gang")) : 3; g > 0; g--) gangs.Add(g.ToString(Py.Inv));
            else gangs.Add("");
            out_.Add(new JsonObject
            {
                ["name"] = J.Clone(J.Truthy(J.Get(r, "name")) ? J.Get(r, "name") : J.Get(r, "rule")),
                ["rule"] = J.Clone(J.Get(r, "rule")),
                ["gangs"] = gangs,
                ["points"] = J.Truthy(J.Get(r, "points")) ? J.Clone(J.Get(r, "points")) : new JsonArray(),
                ["roomTypes"] = J.Truthy(J.Get(r, "room_types")) ? J.Clone(J.Get(r, "room_types")) : new JsonArray(),
            });
        }
        out_.Add(new JsonObject { ["name"] = MatterGateway, ["rule"] = "other_brand", ["gangs"] = new JsonArray(""), ["points"] = new JsonArray(), ["roomTypes"] = new JsonArray() });
        return out_;
    }

    private static string RuleName(JsonNode? rule) => J.Str(J.Truthy(J.Get(rule, "name")) ? J.Get(rule, "name") : J.Get(rule, "rule")) ?? "";

    /// <summary>{gang/"": catalogue items, best first}: the company's products for this rule (catalog["ruleItems"], by
    /// id); a rule may also name items itself (tests, the command line)</summary>
    private static List<(string Gang, List<JsonObject> Items)> Candidates(JsonNode? rule, JsonObject catalog, Dictionary<long, JsonObject> byId,
                                                                         Dictionary<string, JsonObject> byName)
    {
        var name = RuleName(rule);
        var mapped = J.Get(J.Get(catalog, "ruleItems"), name);
        var out_ = new List<(string, List<JsonObject>)>();
        if (mapped != null)
        {
            foreach (var (g, ids) in J.Obj(mapped))
                out_.Add((g, J.Items(ids).Select(J.Int).Where(byId.ContainsKey).Select(i => byId[i]).ToList()));
            return out_;
        }
        var named = J.Get(rule, "items");
        if (named is JsonObject groups)
            foreach (var (g, names) in groups)
                out_.Add((g, J.Strings(names).Where(n => byName.ContainsKey(Common.Norm(n))).Select(n => byName[Common.Norm(n)]).ToList()));
        else
            out_.Add(("", J.Strings(named).Where(n => byName.ContainsKey(Common.Norm(n))).Select(n => byName[Common.Norm(n)]).ToList()));
        return out_;
    }

    /// <summary>Can go in the plan at all (a "both" product still needs confirming for Premium)</summary>
    public static bool Usable(string? usage, string plan) => usage == "both" || usage == plan;

    private static IEnumerable<(JsonNode F, JsonNode A)> Areas(JsonNode extraction)
    {
        foreach (var f in J.Items(J.Get(extraction, "floors")))
            foreach (var a in J.Items(J.Get(f, "areas")))
                yield return (f!, a!);
    }

    private static long ItemCount(JsonNode a, string p) => J.Int(J.Get(J.Get(a, "items"), p));

    private static string Key(JsonNode f, JsonNode a) => $"{J.Str(J.Get(f, "name"))} / {J.Str(J.Get(a, "name"))}";

    private sealed class PerArea : Dictionary<string, Dictionary<string, double>>
    {
        public Dictionary<string, double> At(string gang)
        {
            if (!TryGetValue(gang, out var d)) this[gang] = d = new Dictionary<string, double>();
            return d;
        }
    }

    private static double Opt(JsonNode? rule, string key, double dflt) => J.Has(rule, key) ? (J.Num(J.Get(rule, key)) ?? 0) : dflt;

    /// <summary>(the drawing has what the rule needs, {gang/"": {area: qty}}, explanation)</summary>
    private static (bool Applies, PerArea PerArea, string Why) Derive(JsonNode rule, JsonNode extraction)
    {
        var points = J.Strings(J.Get(rule, "points"));
        var roomTypes = J.Strings(J.Get(rule, "room_types"));
        double mult = Opt(rule, "multiplier", 1);
        double ratio = Opt(rule, "ratio", 1);
        if (ratio == 0) ratio = 1;
        var kind = J.Str(J.Get(rule, "rule"));
        var perArea = new PerArea();
        if (kind == "gangs") return Gangs(rule, extraction);
        if (kind == "per_point")
        {
            long total = 0;
            foreach (var (f, a) in Areas(extraction))
            {
                long n = points.Sum(p => ItemCount(a, p));
                if (n != 0)
                {
                    total += n;
                    perArea.At("")[Key(f, a)] = Ceil(n * mult / ratio);
                }
            }
            double qty = Ceil(total * mult / ratio);
            // the per-area figures are rounded up one by one; keep the total exact
            if (perArea.TryGetValue("", out var all) && all.Count > 0 && all.Values.Sum() != qty)
                perArea[""] = new Dictionary<string, double> { ["All areas"] = qty };
            return (total > 0, perArea, $"{total} {string.Join("/", points)} point(s)" + (mult != 1 ? $" x {Py.G(mult)}" : "") + (ratio != 1 ? $" / {Py.G(ratio)}" : ""));
        }
        if (kind == "per_area")
        {
            int n = 0;
            foreach (var (f, a) in Areas(extraction))
            {
                bool hasPoints = points.Count > 0 && points.Any(p => ItemCount(a, p) > 0);
                bool isRoom = roomTypes.Count > 0 && J.Str(J.Get(a, "roomType")) is { } rt && roomTypes.Contains(rt);
                if (hasPoints || isRoom)
                {
                    n++;
                    perArea.At("")[Key(f, a)] = Ceil(mult);
                }
            }
            var parts = new[] { points.Count > 0 ? $"{string.Join("/", points)} points" : "", string.Join("/", roomTypes) }.Where(s => s.Length > 0);
            return (n > 0, perArea, $"{n} area(s) with {string.Join(" or ", parts)}");
        }
        if (kind == "per_floor")
        {
            var floors = J.Items(J.Get(extraction, "floors")).Where(f => J.Str(J.Get(f, "name")) != "Roof"
                                                                         && J.Items(J.Get(f, "areas")).Any(a => J.Str(J.Get(a, "name")) != "Unassigned")).ToList();
            foreach (var f in floors) perArea.At("")[J.Str(J.Get(f, "name"))!] = Ceil(mult);
            return (floors.Count > 0, perArea, $"{floors.Count} floor(s)");
        }
        return (false, new PerArea(), "unknown rule " + (kind == null ? "None" : $"'{kind}'"));
    }

    /// <summary>light/fan/exhaust points an area without any drawn probably has (room type and size)</summary>
    private static long EstimatedLoads(JsonNode area, JsonNode est)
    {
        var room = J.Str(J.Get(area, "roomType")) is { Length: > 0 } r ? r : "other";
        double sqft = J.Truthy(J.Get(area, "areaSqft")) ? J.Num(J.Get(area, "areaSqft")) ?? 0 : 0;
        long lights;
        if (sqft != 0) lights = (long)Math.Max(1, Ceil(sqft / Opt(est, "sqft_per_light", 60)));
        else
        {
            var table = J.Get(est, "lights");
            lights = J.Has(table, room) ? J.Int(J.Get(table, room)) : J.Has(table, "other") ? J.Int(J.Get(table, "other")) : 2;
        }
        return lights + (J.Strings(J.Get(est, "fan")).Contains(room) ? 1 : 0) + (J.Strings(J.Get(est, "exhaust_fan")).Contains(room) ? 1 : 0);
    }

    /// <summary>Switch boards for every area: the switches drawn there, else its load points, else estimated loads</summary>
    private static (bool, PerArea, string) Gangs(JsonNode rule, JsonNode extraction)
    {
        long top = J.Has(rule, "max_gang") ? J.Int(J.Get(rule, "max_gang")) : 3;
        var loads = J.Strings(J.Get(rule, "points"));
        var switches = J.Strings(J.Get(rule, "switch_points"));
        var est = J.Get(rule, "estimate");
        var perArea = new PerArea();
        var counts = new Dictionary<string, long[]> { ["switch"] = new long[2], ["load"] = new long[2], ["estimate"] = new long[2] };
        foreach (var (f, a) in Areas(extraction))
        {
            long drawnSwitches = switches.Sum(p => ItemCount(a, p));
            long drawnLoads = loads.Sum(p => ItemCount(a, p));
            long n;
            string basis;
            if (drawnSwitches > 0) (n, basis) = (drawnSwitches, "switch");
            else if (drawnLoads > 0) (n, basis) = (drawnLoads, "load");
            else if (est != null && J.Str(J.Get(a, "name")) != "Unassigned" && J.Str(J.Get(f, "name")) != "Roof")
                (n, basis) = (EstimatedLoads(a, est), "estimate");       // no lights/fans drawn
            else continue;
            counts[basis][0] += 1;
            counts[basis][1] += n;
            var key = Key(f, a);
            long full = Math.DivRem(n, top, out long rest);
            if (rest < 0) { rest += top; full -= 1; }      // (Python divmod)
            if (full != 0) perArea.At(top.ToString(Py.Inv))[key] = full;
            if (rest != 0) perArea.At(rest.ToString(Py.Inv))[key] = 1;
        }
        var why = new List<string>();
        if (counts["switch"][0] != 0) why.Add($"{counts["switch"][1]} switch(es) drawn in {counts["switch"][0]} area(s)");
        if (counts["load"][0] != 0) why.Add($"{counts["load"][1]} {string.Join("/", loads)} point(s) in {counts["load"][0]} area(s)");
        if (counts["estimate"][0] != 0) why.Add($"about {counts["estimate"][1]} load(s) estimated for {counts["estimate"][0]} area(s) with no lights/fans drawn");
        return (perArea.Count > 0, perArea, string.Join("; ", why) + $" - on switch boards of up to {top}");
    }

    private static JsonObject AreasNode(Dictionary<string, double> d)
    {
        var o = new JsonObject();
        foreach (var (k, v) in d) o[k] = J.N(v);
        return o;
    }

    private static string? Usage(JsonObject item) => item.ContainsKey("usage") ? J.Str(item["usage"]) : "none";

    private static JsonNode? UsageNode(JsonObject item) => item.ContainsKey("usage") ? J.Clone(item["usage"]) : "none";

    private static string Title(string s) => Py.Title(s);

    /// <summary>The client already uses another company's automation in some areas: one Matter gateway per such area
    /// lets those products work with ours. It is the last row, in every plan. The product is the one the company
    /// mapped to the "Matter gateway" rule; without one the row is still shown, unlinked.</summary>
    private static JsonObject? MatterGatewayLine(JsonNode other, JsonObject catalog, List<JsonObject> lines, List<JsonObject> leftOut, JsonObject? size)
    {
        var areas = J.Items(J.Get(other, "areas")).Select(a => J.PyStr(a).Trim()).Where(a => a.Length > 0).Distinct().ToList();
        if (areas.Count == 0) return null;
        var company = (J.Truthy(J.Get(other, "company")) ? J.PyStr(J.Get(other, "company")) : "").Trim();
        if (company.Length == 0) company = "another company";
        var items = J.Items(J.Get(catalog, "items")).OfType<JsonObject>().ToList();
        var byId = new Dictionary<long, JsonObject>();
        foreach (var i in items) byId[J.Int(i["id"])] = i;
        var mapped = Candidates(new JsonObject { ["name"] = MatterGateway }, catalog, byId, new()).Where(c => c.Gang == "").SelectMany(c => c.Items).ToList();
        var item = mapped.Count > 0 ? mapped[0] : null;
        double packageQty = 0;
        if (item != null)       // already in the package: shown once, at the end
        {
            long id = J.Int(item["id"]);
            foreach (var bucket in new[] { lines, leftOut })
                foreach (var l in bucket.Where(l => J.Int(l["itemId"]) == id).ToList())
                {
                    bucket.Remove(l);
                    if (bucket == lines) packageQty += J.Num(l["quantity"]) ?? 0;
                }
        }
        int n = areas.Count;
        var areaNode = new JsonObject();
        foreach (var a in areas) areaNode[a] = 1;
        return new JsonObject
        {
            ["itemId"] = item != null ? J.Clone(item["id"]) : 0,
            ["name"] = item != null ? J.Clone(item["name"]) : MatterGateway,
            ["groupName"] = item != null ? J.Clone(J.Get(item, "groupName")) : null,
            ["productLineName"] = J.Truthy(J.Get(item, "productLineName")) ? J.Clone(J.Get(item, "productLineName")) : MatterGateway,
            ["usage"] = J.Truthy(J.Get(item, "usage")) ? J.Clone(J.Get(item, "usage")) : "none",
            ["productId"] = item != null ? J.Clone(J.Get(item, "productId")) : null,
            ["productName"] = item != null ? J.Clone(J.Get(item, "productName")) : null,
            ["quantity"] = J.N(n + packageQty),
            ["baseline"] = size != null ? J.N(Qty(J.Get(J.Get(item, "quantities"), J.PyStr(size["id"])))) : 0,
            ["source"] = "other",
            ["detail"] = $"client uses {company} products in {n} area(s) - one Matter gateway per area" + (packageQty != 0 ? $" + {J.PyStr(J.N(packageQty))} from the package" : ""),
            ["areas"] = areaNode,
            ["otherCompany"] = company,
        };
    }

    /// <summary>choices: null = not asked yet; else {item id: "use" | "skip" | id of a Premium-only item to use instead}.
    /// otherBrand: {"company": name, "areas": ["Ground Floor / Lounge", ...]} -> a Matter gateway row (last).
    /// installation: installation charges, for the product breakdown's total</summary>
    public static JsonObject Suggest(JsonObject extraction, JsonObject catalog, string? plan, JsonObject? choices = null, long? sizeId = null,
                                     JsonObject? rules = null, JsonObject? otherBrand = null, double? installation = null)
    {
        plan = (plan ?? "").ToLowerInvariant();
        if (!Plans.Contains(plan)) throw new SolutionException("Choose the Standard or the Premium plan.");
        rules ??= LoadRules();
        var items = J.Items(J.Get(catalog, "items")).OfType<JsonObject>().ToList();
        var byName = new Dictionary<string, JsonObject>();
        foreach (var i in items) byName[Common.Norm(J.Str(i["name"]))] = i;
        var size = BuildingSize.Choose(extraction, J.Items(J.Get(catalog, "sizes")).OfType<JsonObject>().ToList(), J.Obj(J.Get(rules, "building_size")), sizeId);
        var notes = new List<string>();
        string? sizeKey = size != null ? J.PyStr(size["id"]) : null;

        var qty = new Dictionary<long, double>();
        var source = new Dictionary<long, string>();
        var detail = new Dictionary<long, string>();
        var areas = new Dictionary<long, Dictionary<string, double>>();
        var servesSame = new Dictionary<long, PyIntSet>();      // items of one rule serve the same points (Premium alternatives)
        long Id(JsonObject i) => J.Int(i["id"]);
        foreach (var i in items)
        {
            qty[Id(i)] = size != null ? Qty(J.Get(J.Get(i, "quantities"), sizeKey!)) : 0;
            source[Id(i)] = "package";
            detail[Id(i)] = size != null ? $"{J.PyStr(size["name"])} package" : "no package size";
        }

        // the drawing decides the items it has points for
        var byIdAll = new Dictionary<long, JsonObject>();
        foreach (var i in items) byIdAll[Id(i)] = i;
        foreach (var rule in J.Items(J.Get(rules, "rules")))
        {
            var name = RuleName(rule);
            var (applies, perArea, why) = Derive(rule!, extraction);
            var candidates = Candidates(rule, catalog, byIdAll, byName);
            if (!applies)
            {
                notes.Add($"{name}: the drawing shows none of the points this needs - the package quantity is kept.");
                continue;
            }
            foreach (var (gang, found) in candidates)
            {
                foreach (var i in found)
                {
                    if (!servesSame.TryGetValue(Id(i), out var set)) servesSame[Id(i)] = set = new PyIntSet();
                    foreach (var f in found) if (Id(f) != Id(i)) set.Add(Id(f));
                }
                var pick = found.FirstOrDefault(i => Usable(Usage(i), plan));
                var wanted = perArea.TryGetValue(gang, out var w) ? w : new Dictionary<string, double>();
                foreach (var i in found)
                {
                    if (pick != null && Id(i) == Id(pick))
                    {
                        qty[Id(i)] = wanted.Values.Sum();
                        source[Id(i)] = "drawing";
                        detail[Id(i)] = why;
                        areas[Id(i)] = wanted;
                    }
                    else
                    {
                        qty[Id(i)] = 0;           // replaced by the drawing's figure (or not usable in this plan)
                        source[Id(i)] = "drawing";
                        detail[Id(i)] = why + (pick != null ? $" - {J.PyStr(pick["name"])} used" : "");
                    }
                }
                if (pick == null && wanted.Count > 0 && found.Count > 0)
                {
                    var label = gang.Length > 0 ? $"{gang}-gang " : "";
                    var other = plan == "standard" ? "premium" : "standard";
                    var names = string.Join("/", found.Select(i => J.PyStr(i["name"])));
                    if (found.All(i => Usage(i) == other))
                        notes.Add($"{name}: {names} is {Title(other)} only - not in a {Title(plan)} package.");
                    else
                        notes.Add($"{name}: no {Title(plan)} product for {label}{names} - categorize one for " +
                                  $"{Title(plan)} on the Product Categorization tab (needed: {J.PyStr(J.N(wanted.Values.Sum()))}).");
                }
            }
            if (!candidates.Any(c => c.Items.Count > 0))
                notes.Add($"{name}: no product is chosen for this drawing rule - choose them on Solution Packages > Drawing rules.");
        }

        JsonObject Line(JsonObject i, double q, JsonObject? extra = null)
        {
            var o = new JsonObject
            {
                ["itemId"] = J.Clone(i["id"]), ["name"] = J.Clone(i["name"]), ["groupName"] = J.Clone(J.Get(i, "groupName")),
                ["productLineName"] = J.Clone(J.Get(i, "productLineName")), ["usage"] = UsageNode(i),
                ["productId"] = J.Clone(J.Get(i, "productId")), ["productName"] = J.Clone(J.Get(i, "productName")),
                ["quantity"] = J.N(q), ["baseline"] = size != null ? J.N(Qty(J.Get(J.Get(i, "quantities"), sizeKey!))) : 0,
                ["source"] = source[Id(i)], ["detail"] = detail[Id(i)],
                ["areas"] = AreasNode(areas.TryGetValue(Id(i), out var a) ? a : new()),
            };
            if (extra != null)
                foreach (var (k, v) in extra) o[k] = J.Clone(v);
            return o;
        }

        var lines = new List<JsonObject>();
        var toConfirm = new List<JsonObject>();
        var leftOut = new List<JsonObject>();
        var byId = byIdAll;
        Dictionary<string, string>? ch = choices?.ToDictionary(kv => kv.Key, kv => J.PyStr(kv.Value));
        var added = new List<(long Id, double Q)>();      // Premium-only items chosen instead of a "both" product
        foreach (var i in items)
        {
            double q = qty[Id(i)];
            if (q <= 0) continue;
            var usage = Usage(i);
            if (usage == "none") leftOut.Add(Line(i, q, new JsonObject { ["reason"] = "not categorised" }));
            else if (plan == "standard")
            {
                if (usage is "standard" or "both") lines.Add(Line(i, q));
                else leftOut.Add(Line(i, q, usage == "premium" ? new JsonObject { ["reason"] = "Premium only" } : null));
            }
            else if (usage == "premium") lines.Add(Line(i, q));
            else if (usage == "standard") leftOut.Add(Line(i, q, new JsonObject { ["reason"] = "Standard only" }));
            else
            {
                // both, Premium plan: Premium-only items that serve the same points (same drawing rule); for
                // package-only items the same group
                List<JsonObject> same;
                if (servesSame.TryGetValue(Id(i), out var set)) same = set.Items().Select(x => byId[x]).ToList();
                else
                {
                    var group = J.Get(i, "groupName");
                    same = items.Where(a => !servesSame.ContainsKey(Id(a)) && J.Truthy(J.Get(a, "groupName"))
                                            && J.Get(a, "groupName")!.ToJsonString() == group?.ToJsonString()).ToList();
                }
                var alternatives = new JsonArray(same.Where(a => Usage(a) == "premium")
                                                     .Select(a => (JsonNode)new JsonObject { ["itemId"] = J.Clone(a["id"]), ["name"] = J.Clone(a["name"]) }).ToArray());
                if (alternatives.Count == 0)            // nothing Premium-only to swap it for: just use it, don't ask
                {
                    lines.Add(Line(i, q));
                    continue;
                }
                if (ch == null)
                {
                    toConfirm.Add(Line(i, q, new JsonObject { ["alternatives"] = alternatives }));
                    continue;
                }
                var choice = ch.TryGetValue(J.PyStr(i["id"]), out var c) ? c : "use";
                if (choice == "use") lines.Add(Line(i, q, new JsonObject { ["confirmed"] = true }));
                else if (choice.Length > 0 && choice.All(char.IsDigit) && long.TryParse(choice, out var alt) && byId.TryGetValue(alt, out var altItem)
                         && Usage(altItem) == "premium")
                {
                    int k = added.FindIndex(x => x.Id == alt);
                    if (k >= 0) added[k] = (alt, added[k].Q + q);
                    else added.Add((alt, q));
                    leftOut.Add(Line(i, q, new JsonObject { ["reason"] = $"replaced by {J.PyStr(altItem["name"])}" }));
                }
                else leftOut.Add(Line(i, q, new JsonObject { ["reason"] = "you chose not to use it" }));
            }
        }
        foreach (var (altId, q) in added)
        {
            var existing = lines.FirstOrDefault(l => J.Int(l["itemId"]) == altId);
            if (existing != null)
            {
                existing["quantity"] = J.N((J.Num(existing["quantity"]) ?? 0) + q);
                existing["detail"] = J.Str(existing["detail"]) + $" + {J.PyStr(J.N(q))} instead of a Standard/Premium product";
            }
            else
            {
                source[altId] = "choice";
                detail[altId] = "chosen instead of a product ticked for both";
                lines.Add(Line(byId[altId], q));
            }
        }

        var order = new Dictionary<long, int>();
        for (int n = 0; n < items.Count; n++) order[Id(items[n])] = n;
        lines = lines.Select((l, k) => (l, k)).OrderBy(x => order.TryGetValue(J.Int(x.l["itemId"]), out var o) ? o : 0).ThenBy(x => x.k).Select(x => x.l).ToList();
        var gateway = otherBrand != null && otherBrand.Count > 0 && toConfirm.Count == 0 ? MatterGatewayLine(otherBrand, catalog, lines, leftOut, size) : null;
        if (gateway != null)
        {
            lines.Add(gateway);
            if (J.Int(gateway["itemId"]) == 0)
                notes.Add($"{MatterGateway}: choose its product on Solution Packages > Drawing rules, so quotations include it.");
        }
        var unlinked = lines.Where(l => l["productId"] == null).ToList();
        if (unlinked.Count > 0 && toConfirm.Count == 0)
            notes.Add($"{unlinked.Count} item(s) are not in the inventory (not on the price list) - their price is entered on the quotation.");
        bool asking = toConfirm.Count > 0;
        var result = new JsonObject
        {
            ["plan"] = plan,
            ["size"] = size,
            ["needsConfirmation"] = asking,
            ["toConfirm"] = new JsonArray(toConfirm.Cast<JsonNode>().ToArray()),
            ["lines"] = asking ? new JsonArray() : new JsonArray(lines.Cast<JsonNode>().ToArray()),
            ["leftOut"] = new JsonArray(leftOut.Cast<JsonNode>().ToArray()),
            ["totals"] = new JsonObject
            {
                ["items"] = asking ? 0 : lines.Count,
                ["quantity"] = asking ? 0 : J.N(lines.Sum(l => J.Num(l["quantity"]) ?? 0)),
                ["unlinked"] = asking ? 0 : unlinked.Count,
            },
            ["notes"] = new JsonArray(notes.Select(s => (JsonNode)s).ToArray()),
        };
        // the products per area, floor by floor (the quotation sheet layout) - once the lines are settled
        result["breakdown"] = asking ? null : Breakdown.Of(result, extraction, catalog, installation: installation);
        return result;
    }
}

/// <summary>Which package building size (1 BHK ... 10 Kanal) a drawing is: from the size written on it, else from its
/// rooms (bedrooms for a small single-floor flat, floor area -> plot size for a house).</summary>
public static class BuildingSize
{
    private static readonly Regex Size = new(@"(\d+(?:\.\d+)?)\s*(MARLA|KANAL|BHK)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    public static (string Kind, double Value)? Parse(string? name)
    {
        var m = Size.Match(name ?? "");
        return m.Success ? (m.Groups[2].Value.ToUpperInvariant(), double.Parse(m.Groups[1].Value, Py.Inv)) : null;
    }

    private static double Cfg(JsonObject cfg, string key, double dflt) => cfg.ContainsKey(key) ? J.Num(cfg[key]) ?? 0 : dflt;

    public static double PlotSqft(string kind, double value, JsonObject cfg)
    {
        double marla = Cfg(cfg, "sqft_per_marla", 225);
        return value * marla * (kind == "KANAL" ? Cfg(cfg, "marla_per_kanal", 20) : 1);
    }

    /// <summary>floors with rooms (not the roof), the largest floor's room area, bedrooms</summary>
    private static (int Floors, double Largest, int Bedrooms) FloorStats(JsonObject extraction)
    {
        int floors = 0, bedrooms = 0;
        double largest = 0;
        foreach (var f in J.Items(J.Get(extraction, "floors")))
        {
            var rooms = J.Items(J.Get(f, "areas")).Where(a => J.Str(J.Get(a, "name")) != "Unassigned").ToList();
            if (rooms.Count == 0 || J.Str(J.Get(f, "name")) == "Roof") continue;
            floors++;
            double sum = 0;
            foreach (var a in rooms) sum += J.Truthy(J.Get(a, "areaSqft")) ? J.Num(J.Get(a, "areaSqft")) ?? 0 : 0;
            largest = Math.Max(largest, sum);
            bedrooms += rooms.Count(a => J.Str(J.Get(a, "roomType")) == "bedroom");
        }
        return (floors, largest, bedrooms);
    }

    private static JsonObject Pick(JsonObject s, string reason) => new() { ["id"] = J.Clone(s["id"]), ["name"] = J.Clone(s["name"]), ["reason"] = reason };

    public static JsonObject? Choose(JsonObject extraction, List<JsonObject> sizes, JsonObject cfg, long? overrideId = null)
    {
        if (sizes.Count == 0) return null;
        if (overrideId is { } oid && oid != 0)
        {
            var s = sizes.FirstOrDefault(x => J.Num(x["id"]) == oid);
            if (s != null) return Pick(s, "chosen by you");
        }
        var parsed = sizes.Select(s => (S: s, P: Parse(J.Str(s["name"])))).ToList();
        var houses = parsed.Where(x => x.P is { Kind: "MARLA" or "KANAL" }).Select(x => (x.S, P: x.P!.Value)).ToList();
        var flats = parsed.Where(x => x.P is { Kind: "BHK" }).Select(x => (x.S, P: x.P!.Value)).ToList();

        // 1) the size written on the drawing ("10 MARLA" in the title block)
        var plotSize = J.Get(J.Get(extraction, "project"), "plotSize");
        var plotText = J.Truthy(plotSize) ? J.PyStr(plotSize) : "";
        var written = Parse(plotText);
        if (written is { } wr)
        {
            var same = parsed.Where(x => x.P is { } p && p.Kind == wr.Kind && p.Value == wr.Value).ToList();
            if (same.Count > 0) return Pick(same[0].S, $"the drawing says {plotText}");
            if (wr.Kind == "BHK" && flats.Count > 0)
                return Pick(Py.MinBy(flats, x => Math.Abs(x.P.Value - wr.Value)).S, $"the drawing says {plotText} - nearest size");
            if (houses.Count > 0)
            {
                double want = PlotSqft(wr.Kind, wr.Value, cfg);
                return Pick(Py.MinBy(houses, x => Math.Abs(Math.Log(PlotSqft(x.P.Kind, x.P.Value, cfg) / want))).S, $"the drawing says {plotText} - nearest size");
            }
        }

        var (floors, largest, bedrooms) = FloorStats(extraction);
        // 2) a small single-floor plan with bedrooms: a flat
        if (flats.Count > 0 && bedrooms != 0 && ((floors <= 1 && 0 < largest && largest <= Cfg(cfg, "bhk_max_sqft", 1800)) || houses.Count == 0))
            return Pick(Py.MinBy(flats, x => Math.Abs(x.P.Value - bedrooms)).S, $"{bedrooms} bedroom(s) on a {Py.RoundInt(largest)} sq ft floor");
        // 3) a house: plot size from the largest floor's area
        if (houses.Count > 0 && largest > 0)
        {
            double plot = largest / Cfg(cfg, "floor_coverage", 0.8);
            var s = Py.MinBy(houses, x => Math.Abs(Math.Log(PlotSqft(x.P.Kind, x.P.Value, cfg) / plot))).S;
            return Pick(s, $"largest floor {Py.RoundInt(largest)} sq ft = about {Py.RoundInt(plot)} sq ft plot " +
                           $"({Py.Repr(Py.Round(plot / Cfg(cfg, "sqft_per_marla", 225), 1))} Marla)");
        }
        var mid = Pick(sizes[sizes.Count / 2], "the drawing has no room areas - choose the size yourself");
        mid["uncertain"] = true;
        return mid;
    }
}

/// <summary>The solution as a product breakdown per area (the layout of the quotation sheets: a row per product with
/// its unit price, quantity and total, and a column per area grouped under its floor).</summary>
public static class Breakdown
{
    public const string WholeFloor = "Whole floor";

    public static JsonObject Of(JsonObject result, JsonObject extraction, JsonObject? catalog = null, string? client = null, double? installation = null)
    {
        var prices = new Dictionary<long, JsonNode?>();
        foreach (var i in J.Items(J.Get(catalog, "items")).OfType<JsonObject>()) prices[J.Int(i["id"])] = J.Get(i, "unitPrice");
        var lines = J.Items(J.Get(result, "lines")).OfType<JsonObject>().Where(l => J.Truthy(J.Get(l, "quantity"))).ToList();

        // the columns: every area of the drawing that a product goes to, floor by floor in the drawing's order
        var used = lines.SelectMany(l => J.Obj(J.Get(l, "areas")).Select(kv => kv.Key)).ToHashSet();
        var floors = new JsonArray();
        var keys = new List<(string Floor, string Area)>();
        foreach (var f in J.Items(J.Get(extraction, "floors")))
        {
            var fname = J.Str(J.Get(f, "name")) ?? "";
            var cols = J.Items(J.Get(f, "areas")).Select(a => J.Str(J.Get(a, "name")) ?? "").Where(a => used.Contains($"{fname} / {a}")).ToList();
            if (used.Contains(fname)) cols.Add(WholeFloor);
            if (cols.Count > 0)
            {
                floors.Add(new JsonObject { ["name"] = fname, ["areas"] = new JsonArray(cols.Select(c => (JsonNode)c).ToArray()) });
                keys.AddRange(cols.Select(c => (fname, c)));
            }
        }

        var rows = new JsonArray();
        double subtotal = 0;
        bool priced = true;
        foreach (var l in lines)
        {
            var per = J.Obj(J.Get(l, "areas"));
            var cells = new JsonArray();
            double inAreas = 0;
            foreach (var (f, a) in keys)
            {
                var v = J.Get(per, a == WholeFloor ? f : $"{f} / {a}");
                if (J.Truthy(v))
                {
                    cells.Add(J.Clone(v));
                    inAreas += J.Num(v) ?? 0;
                }
                else cells.Add(null);
            }
            double qty = J.Num(l["quantity"]) ?? 0;
            var priceNode = l.ContainsKey("unitPrice") ? l["unitPrice"] : prices.TryGetValue(J.Int(J.Get(l, "itemId")), out var p) ? p : null;
            double? price = J.Num(priceNode);
            double? total = price != null ? Py.Round(price.Value * qty, 2) : null;
            if (price == null) priced = false;
            subtotal += total ?? 0;
            rows.Add(new JsonObject
            {
                ["itemId"] = J.Clone(J.Get(l, "itemId")),
                ["product"] = J.Truthy(J.Get(l, "productName")) ? J.Clone(l["productName"]) : J.Clone(l["name"]),
                ["unitPrice"] = J.Clone(priceNode),
                ["quantity"] = J.Clone(l["quantity"]),
                ["total"] = total != null ? J.N(total.Value) : null,
                // (all of it in the areas listed - else part is the package's, kept in Qty only)
                ["byArea"] = cells,
                ["inAreas"] = inAreas == qty,
            });
        }
        subtotal = Py.Round(subtotal, 2);
        var proj = J.Get(extraction, "project");
        string clientName = client ?? (J.Truthy(J.Get(proj, "client")) ? J.PyStr(J.Get(proj, "client")) : J.Truthy(J.Get(extraction, "fileName")) ? J.PyStr(J.Get(extraction, "fileName")) : "");
        return new JsonObject
        {
            ["client"] = clientName,
            ["plan"] = J.Clone(J.Get(result, "plan")),
            ["floors"] = floors,
            ["rows"] = rows,
            ["subtotal"] = J.N(subtotal),
            ["installation"] = installation is { } inst ? J.N(inst) : null,
            ["total"] = J.N(Py.Round(subtotal + (installation ?? 0), 2)),
            ["priced"] = priced,
        };
    }
}
