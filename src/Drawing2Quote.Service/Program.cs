// Drawing2Quote service (.NET), called by the POS backend - the same API as the Python service (branch main):
//   GET  /health                 is it running (and which reader: "dotnet")
//   POST /extract   (file)       validate + read the drawing -> floors / areas / point counts (waits for the result)
//   POST /jobs      (file)       the same in the background -> {jobId}; refused files answer 422 straight away
//   GET  /jobs/{id}              {status: queued|running|done|failed, page, pages, message, result | detail}
//   GET  /rules                  the drawing rules a company chooses its products for
//   POST /suggest   (json)       {extraction, catalog, plan, choices?, sizeId?, otherBrand?, installation?, client?}
//   POST /suggest/xlsx (json)    the same -> the product breakdown as an Excel sheet
// Settings (appsettings.{Environment}.json, "Reader"): Key - the X-Service-Key the POS must send (D2Q_KEY overrides);
// AllowedOrigins - browser origins allowed to call it (the POS frontends).
using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Drawing2Quote.Reader;
using Drawing2Quote.Service.Rules;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);
const long uploadLimit = Validator.MaxBytes + 4 * 1024 * 1024;     // a drawing up to 200 MB plus the form around it
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = uploadLimit);
builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = uploadLimit);
var origins = builder.Configuration.GetSection("Reader:AllowedOrigins").Get<string[]>() ?? Array.Empty<string>();
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p.WithOrigins(origins).AllowAnyHeader().AllowAnyMethod()));
var app = builder.Build();
app.UseCors();

string? serviceKey = Environment.GetEnvironmentVariable("D2Q_KEY") is { Length: > 0 } envKey ? envKey : app.Configuration["Reader:Key"];
var log = app.Logger;

// a file refused for what it is (a DWG, a locked PDF ...) is the user's to fix; one refused because the reader failed
// inside is a bug to find: its cause and stack trace go to the log (the console, and the Windows event log)
void LogIfInternal(RejectedFileException exc, string file)
{
    if (exc.InnerException is { } cause) log.LogError(cause, "Reading {File} failed inside the reader", file);
}
log.LogInformation("Drawing2Quote (.NET) - {Environment}; service key {Key}; browser origins: {Origins}",
                   app.Environment.EnvironmentName, string.IsNullOrEmpty(serviceKey) ? "not required" : "required", string.Join(", ", origins));

static IResult Detail(int status, string message) => Results.Json(new { detail = message }, statusCode: status);

static IResult JsonText(string json) => Results.Text(json, "application/json; charset=utf-8");

// every endpoint but /health needs the service key (when one is set)
app.Use(async (ctx, next) =>
{
    if (!string.IsNullOrEmpty(serviceKey) && ctx.Request.Path != "/health" && !HttpMethods.IsOptions(ctx.Request.Method)
        && ctx.Request.Headers["X-Service-Key"] != serviceKey)
    {
        await Detail(401, "Wrong service key.").ExecuteAsync(ctx);
        return;
    }
    await next();
});

app.MapGet("/health", () => Results.Json(new { status = "ok", supported = Validator.SupportedMessage, reader = "dotnet" }));

static async Task<(string Name, byte[]? Data)> ReadUpload(HttpRequest req)
{
    if (!req.HasFormContentType) return ("", null);
    var form = await req.ReadFormAsync();
    var file = form.Files.GetFile("file");
    if (file == null) return ("", null);
    using var s = file.OpenReadStream();
    using var ms = new MemoryStream();
    var buf = new byte[81920];
    int n;
    while ((n = await s.ReadAsync(buf)) > 0)          // (read up to just over the limit: the check names the size)
    {
        ms.Write(buf, 0, n);
        if (ms.Length > Validator.MaxBytes) break;
    }
    return (file.FileName ?? "", ms.ToArray());
}

app.MapPost("/extract", async (HttpRequest req) =>
{
    var (name, data) = await ReadUpload(req);
    if (data == null) return Detail(422, "Choose the drawing file.");
    try
    {
        var ex = await Task.Run(() => DrawingReader.Extract(name, data));
        return JsonText(ex.ToJson(indented: false));
    }
    catch (RejectedFileException exc)
    {
        LogIfInternal(exc, name);
        return Detail(422, exc.Message);
    }
});

// ── background reading: a printed PDF is read with OCR (about a minute per sheet), far longer than a request may take
var jobs = new ConcurrentDictionary<string, Job>();
var oneAtATime = new SemaphoreSlim(1, 1);       // one drawing at a time, as the Python service
var keepFor = TimeSpan.FromHours(2);

app.MapPost("/jobs", async (HttpRequest req) =>
{
    var (name, data) = await ReadUpload(req);
    if (data == null) return Detail(422, "Choose the drawing file.");
    try
    {
        Validator.Check(name, data);    // quick checks first: a refused file is answered now, not after the queue
    }
    catch (RejectedFileException exc)
    {
        return Detail(422, exc.Message);
    }
    var now = DateTime.UtcNow;
    foreach (var (k, j) in jobs)
        if (j.Finished is { } f && now - f > keepFor) jobs.TryRemove(k, out _);
    int ahead = jobs.Values.Count(j => j.Status is "queued" or "running");
    var id = Guid.NewGuid().ToString("N");
    var job = new Job { FileName = name, Message = ahead > 0 ? $"Waiting for {ahead} other drawing(s) to be read" : "Starting" };
    jobs[id] = job;
    _ = Task.Run(async () =>
    {
        await oneAtATime.WaitAsync();
        try
        {
            job.Status = "running";
            job.Message = "Reading the drawing";
            Ocr.Progress = (page, pages) =>
            {
                job.Page = page;
                job.Pages = pages;
                job.Message = $"Reading page {page} of {pages} - it was printed to PDF, so its words are read " +
                              "from the page image (about a minute per sheet; a drawing already read is quick)";
            };
            try
            {
                var ex = DrawingReader.Extract(name, data);
                job.Result = JsonNode.Parse(ex.ToJson(indented: false));
                job.Status = "done";
                job.Message = "Done";
            }
            catch (RejectedFileException exc)
            {
                LogIfInternal(exc, name);
                job.Detail = exc.Message;
                job.Status = "failed";
            }
            catch (Exception exc)       // never leave a job "running" forever
            {
                log.LogError(exc, "Reading {File} failed", name);
                job.Detail = $"The drawing could not be read: {exc.Message}";
                job.Status = "failed";
            }
        }
        finally
        {
            Ocr.Progress = null;
            job.Finished = DateTime.UtcNow;
            oneAtATime.Release();
        }
    });
    return Results.Json(new { jobId = id });
});

app.MapGet("/jobs/{jobId}", (string jobId) =>
    jobs.TryGetValue(jobId, out var job)
        ? JsonText(job.ToJson())
        : Detail(404, "This reading is no longer available - upload the drawing again."));

app.MapGet("/rules", () => JsonText(Solution.RuleList().ToJsonString()));

static (JsonObject? Result, IResult? Error) RunSuggest(JsonObject body)
{
    if (body["extraction"] is not JsonObject extraction || body["catalog"] is not JsonObject catalog || J.Str(body["plan"]) is not { } plan)
        return (null, Detail(422, "extraction, catalog and plan are needed."));
    try
    {
        var result = Solution.Suggest(extraction, catalog, plan, body["choices"] as JsonObject,
                                      J.Num(body["sizeId"]) is { } sid ? (long)sid : null, otherBrand: body["otherBrand"] as JsonObject,
                                      installation: J.Num(body["installation"]));
        if (J.Truthy(body["client"]) && result["breakdown"] is JsonObject b) b["client"] = J.Str(body["client"]);
        return (result, null);
    }
    catch (SolutionException exc)
    {
        return (null, Detail(422, exc.Message));
    }
}

static async Task<JsonObject?> Body(HttpRequest req)
{
    try
    {
        return await JsonNode.ParseAsync(req.Body) as JsonObject;
    }
    catch (JsonException)
    {
        return null;
    }
}

app.MapPost("/suggest", async (HttpRequest req) =>
{
    if (await Body(req) is not { } body) return Detail(422, "The request is not JSON.");
    var (result, error) = RunSuggest(body);
    return error ?? JsonText(result!.ToJsonString());
});

app.MapPost("/suggest/xlsx", async (HttpRequest req) =>
{
    if (await Body(req) is not { } body) return Detail(422, "The request is not JSON.");
    var (result, error) = RunSuggest(body);
    if (error != null) return error;
    if (J.Truthy(result!["needsConfirmation"]))
        return Detail(409, "Confirm the products ticked for both Standard and Premium first.");
    var b = (JsonObject)result["breakdown"]!;
    var client = J.Truthy(b["client"]) ? J.Str(b["client"])! : "solution";
    var name = Regex.Replace(client, @"[^A-Za-z0-9 ._-]+", "").Trim();
    if (name.Length == 0) name = "solution";
    return Results.File(Xlsx.Of(b), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        $"{name} - {J.Str(result["plan"])} solution.xlsx");
});

app.Run();

/// <summary>A drawing being read in the background</summary>
internal sealed class Job
{
    public volatile string Status = "queued";
    public int Page, Pages;
    public string FileName = "";
    public volatile string Message = "";
    public JsonNode? Result;
    public string? Detail;
    public DateTime? Finished;

    /// <summary>{status, page, pages, fileName, message, result | detail} - as the Python service answers</summary>
    public string ToJson()
    {
        var o = new JsonObject { ["status"] = Status, ["page"] = Page, ["pages"] = Pages, ["fileName"] = FileName, ["message"] = Message };
        if (Result != null) o["result"] = Result.DeepClone();
        if (Detail != null) o["detail"] = Detail;
        return o.ToJsonString();
    }
}
