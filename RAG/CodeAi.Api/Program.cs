using CodeAi.Shared.Embeddings;
using CodeAi.Shared.Models;
using CodeAi.Shared.Search;
using Elastic.Clients.Elasticsearch;
using Elastic.Clients.Elasticsearch.QueryDsl;
using Microsoft.OpenApi.Models;
using System.Collections.ObjectModel;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

var esUrl = builder.Configuration["Elasticsearch:Url"] ?? "http://localhost:9200";
var index = builder.Configuration["Elasticsearch:Index"] ?? "code_chunks";
var ollamaUrl = builder.Configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
var embedModel = builder.Configuration["Ollama:EmbedModel"] ?? "nomic-embed-text";
var genModel = builder.Configuration["Ollama:GenModel"] ?? "llama3.1:8b";

builder.Services.AddSingleton(_ => EsFactory.Create(esUrl));
builder.Services.AddHttpClient<OllamaClient>(c => {
    c.BaseAddress = new Uri(ollamaUrl);
    c.Timeout = TimeSpan.FromDays(1); // Ollama can take a while
});
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c => c.SwaggerDoc("v1", new OpenApiInfo { Title = "Code QA", Version = "v1" }));

var app = builder.Build();
app.UseSwagger();
app.UseSwaggerUI();

app.MapPost("/ask", async (AskRequest req, ElasticsearchClient es, OllamaClient ollama, CancellationToken ct) =>
{
    var qvec = await ollama.EmbedAsync("query: " + req.Question, ct);
    var extractPrompt = $@"
You are extracting a retrieval hint for Elasticsearch.

Indexed docs have fields:
- text: code or prose chunk
- symbol: optional exact identifier (class/func/event/topic), case-sensitive
- kind: code|doc|infra|proto|openapi
- service: microservice name (folder)

Task:
1) If the question clearly targets ONE code/event identifier (e.g., PriceCalculatedV6, Foo.Bar, DoWorkAsync, TIMESHEET_APPROVED, UserCreatedV2), return it as `ident`.
   - Prefer PascalCase/CamelCase/snake_case tokens that look like real symbols.
   - Prefer event names like *SomethingV<number>* if present.
   - Include namespace if the question clearly implies it (optional).
2) If there is no single clear identifier (e.g., broad domain Q like ""timesheets""), return `ident: null` and provide 1–4 short keywords in `keywords` that best target the concept in code (e.g., [""timesheet"",""approval"",""overtime""]).
3) Optionally include a `service_hint` array ONLY if the question strongly implies a microservice name you use.

Output JSON ONLY on one line, no prose:
{{
  ""ident"": string|null,
  ""keywords"": string[]|null,
  ""service_hint"": string[]|null
}}

Question:
{req.Question}
";
    var hintJson = await ollama.GenerateAsync(extractPrompt, ct);
    var hint = JsonSerializer.Deserialize<Hint>(hintJson, new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    });
    if (hint is null)
        return Results.BadRequest("Failed to extract hint.");

    // Merge service hints from question and extraction
    var serviceHint = (hint.ServiceHint ?? Array.Empty<string>())
        .Concat(req.ServiceHint ?? Array.Empty<string>())
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToArray();

    Query? filter = null;
    if (serviceHint.Any())
    {
        filter = new TermsQuery
        {
            Field = "service",
            Term = new TermsQueryField(
                new ReadOnlyCollection<FieldValue>(serviceHint.Select(FieldValue.String).ToList())
            )
        };
    }

    // 3) Build query clauses
    var shouldClauses = new List<Action<QueryDescriptor<ChunkDocument>>>();

    if (!string.IsNullOrWhiteSpace(hint.Ident))
    {
        // Exact symbol match (very high boost)
        shouldClauses.Add(sq => sq.Term(t => t
            .Field(f => f.Symbol)
            .Value(hint.Ident)
            .Boost(10)
        ));

        // Exact phrase match in text
        shouldClauses.Add(sq => sq.MatchPhrase(mp => mp
            .Field(f => f.Text)
            .Query(hint.Ident)
            .Boost(6)
        ));
    }
    else if (hint.Keywords?.Any() == true)
    {
        // Keywords broad match
        shouldClauses.Add(sq => sq.MultiMatch(mm => mm
            .Fields("text")
            .Query(string.Join(" ", hint.Keywords))
            .Type(TextQueryType.BestFields)
            .Boost(3)
        ));
    }

    // Always include natural language question match
    shouldClauses.Add(sq => sq.MultiMatch(mm => mm
        .Fields("text")
        .Query(req.Question)
        .Type(TextQueryType.BestFields)
        .Boost(2)
    ));

    // Add semantic similarity via ScriptScore
    shouldClauses.Add(sq => sq.ScriptScore(ss => ss
        .Query(kq => kq.MatchAll(ma => { }))
        .Script(sc => sc
            .Source("cosineSimilarity(params.query_vector, 'vec') + 1.0")
            .Params(p => p.Add("query_vector", qvec))
        )
    ));

    // 4) Execute ES search
    var response = await es.SearchAsync<ChunkDocument>(s => s
        .Index(index)
        .Size(200)
        .Query(q => q.Bool(b => b
            .Should(shouldClauses.ToArray())
            .Filter(filter is null ? null : new[] { filter })
        ))
    , ct);

    // 5) Build hits list
    var hits = response.Hits
        .Select(h => new SearchHit
        {
            Repo = h.Source.Repo,
            Service = h.Source.Service,
            FilePath = h.Source.FilePath,
            Kind = h.Source.Kind,
            Text = h.Source.Text,
            Score = h.Score ?? 0
        })
        .OrderByDescending(h => h.Kind.Equals("code", StringComparison.OrdinalIgnoreCase) ? 1 : 0)
        .ThenByDescending(h => h.Score)
        .GroupBy(h => h.FilePath)
        .SelectMany(g => g)
        .ToList();

    // 6) Build LLM context
    var ctx = string.Join("\n\n", hits.Select((c, i) =>
        $"[{i + 1}] ({c.Service}) {c.Repo}/{c.FilePath} [{c.Kind}]\n{c.Text}"));

    // 7) Answer prompt
    var answerPrompt = $"""
You are a codebase assistant. Use ONLY the provided context.
Ignore unrelated context.
Return a concise answer.

Question:
{req.Question}

Context:
{ctx}
""";

    var answer = await ollama.GenerateAsync(answerPrompt, ct);

    return Results.Ok(new
    {
        ident_used = hint.Ident,
        keywords_used = hint.Keywords,
        services_scoped = serviceHint,
        context_files = hits.Select(c => $"{c.Service}:{c.Repo}/{c.FilePath}"),
        answer
    });
})
.WithOpenApi();

app.Run();

record AskRequest(string Question, string[]? ServiceHint);
record Hint(string? Ident, string[]? Keywords, string[]? ServiceHint);
record SearchHit
{
    public string Repo { get; init; } = "";
    public string Service { get; init; } = "";
    public string FilePath { get; init; } = "";
    public string Kind { get; init; } = "";
    public string Text { get; init; } = "";
    public double Score { get; init; }
}
