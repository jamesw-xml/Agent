using System.CommandLine;
using System.Security.Cryptography;
using System.Text;
using CodeAi.Shared.Embeddings;
using CodeAi.Shared.Models;
using CodeAi.Shared.Search;
using Elastic.Clients.Elasticsearch;
using Microsoft.Extensions.Configuration;
using SharpToken;
using System.Collections.Concurrent;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: false)
    .Build();

string codeRoot = config["CodeRoot"] ?? @"C:\code";
string esUrl = config["Elasticsearch:Url"] ?? "http://localhost:9200";
string indexName = config["Elasticsearch:Index"] ?? "code_chunks";
string ollamaUrl = config["Ollama:BaseUrl"] ?? "http://localhost:11434";
string embedModel = config["Ollama:EmbedModel"] ?? "nomic-embed-text";
int maxSize = int.TryParse(config["MaxFileSizeBytes"], out var m) ? m : 2_000_000;
int size = int.TryParse(config["ChunkSizeTokens"], out var s) ? s : 400;
int step = int.TryParse(config["ChunkStepTokens"], out var st) ? st : 240;
var exts = config.GetSection("IncludeExts").Get<string[]>() ?? Array.Empty<string>();
var ignoreDirectories = config.GetSection("IgnoreDirectories").Get<string[]>() ?? Array.Empty<string>();
var skipNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
{
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock"
};
var es = EsFactory.Create(esUrl);
EsFactory.CreateIndexIfNotExists(es, indexName);
var http = new HttpClient { BaseAddress = new Uri(ollamaUrl) };
var ollama = new OllamaClient(http, embedModel);

var repoOption = new Option<string?>("--repo", "Limit to a single repo folder name under CodeRoot");
var cmd = new RootCommand("Code indexer");
cmd.AddOption(repoOption);

cmd.SetHandler(async (repoFilter) =>
{
    // 1. Gather all files
    var repos = Directory.GetDirectories(codeRoot)
        .Where(d => repoFilter == null || Path.GetFileName(d).Equals(repoFilter, StringComparison.OrdinalIgnoreCase))
        .OrderBy(d => d);

    var allChunks = new ConcurrentBag<(string repo, string repoPath, string file, string chunk, int idx)>();
    Console.WriteLine($"Found {repos.Count()} repos to process.");
    await Parallel.ForEachAsync(repos, new ParallelOptions { MaxDegreeOfParallelism = 32 }, async (repoPath, _) =>
    {
        if (!repoPath.ToLower().Contains("remundo")) return;
        var repo = Path.GetFileName(repoPath);

        var files = Directory.EnumerateFiles(repoPath, "*.*", SearchOption.AllDirectories)
            .Where(p =>
            {
                if (ignoreDirectories.Any(ign =>
                    p.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                     .Any(seg => seg.Equals(ign, StringComparison.OrdinalIgnoreCase))))
                    return false;
                if (skipNames.Contains(Path.GetFileName(p)))
                    return false;
                if (!exts.Contains(Path.GetExtension(p), StringComparer.OrdinalIgnoreCase))
                    return false;
                return true;
            });

        foreach (var file in files)
        {
            var fi = new FileInfo(file);
            if (fi.Length > maxSize) continue;
            if (!TryReadText(file, out var text)) continue;

            var chunks = ChunkTextTokenAware(text, size, step);
            for (int i = 0; i < chunks.Count; i++)
                allChunks.Add((repo, repoPath, file, chunks[i], i));
        }
    });
    var totalSize = allChunks.Sum(c => c.chunk.Length);
    Console.WriteLine($"Total chunks to embed: {totalSize}");

    // 2. Embed all chunks in a global parallel pool
    var results = new ConcurrentBag<(string repo, string repoPath, string file, int idx, float[] vec, string chunk)>();
    int completed = 0; // shared counter

    await Parallel.ForEachAsync(allChunks, new ParallelOptions { MaxDegreeOfParallelism = 32 }, async (item, _) =>
    {
        try
        {
            var vec = await ollama.EmbedAsync(item.chunk);
            var done = Interlocked.Increment(ref completed); // thread-safe increment
            Console.WriteLine($"[{done}/{allChunks.Count}] {item.file}");

            if (vec is { Length: 1024 })
                results.Add((item.repo, item.repoPath, item.file, item.idx, vec, item.chunk));
            else
                Console.WriteLine($"[WARN] {item.file}: invalid vector length");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] {item.file}: {ex.Message}");
        }
    });

    allChunks.Clear();

    // 3. Group back by file and bulk index into ES
    foreach (var group in results.GroupBy(r => r.file))
    {
        var first = group.First();
        var repo = first.repo;
        var repoPath = first.repoPath;
        var filePath = Path.GetRelativePath(repoPath, first.file).Replace("\\", "/");

        var bulkResponse = await es.BulkAsync(b =>
        {
            foreach (var r in group)
            {
                var doc = new ChunkDocument
                {
                    Repo = repo,
                    Service = repo,
                    FilePath = filePath,
                    Language = Path.GetExtension(filePath).Trim('.'),
                    Kind = GuessKind(filePath),
                    Text = r.chunk,
                    Vec = r.vec,
                    Links = Array.Empty<string>()
                };

                b.Index<ChunkDocument>(doc, op => op
                    .Index(indexName)
                    .Id(DocId(repo, doc.FilePath, r.idx)));
            }
        });

        if (bulkResponse.Errors)
            Console.WriteLine($"[WARN] Bulk errors for {repo}/{Path.GetFileName(filePath)}");
        else if (!bulkResponse.IsValidResponse)
            Console.WriteLine($"[ERROR] Failed to index {repo}/{Path.GetFileName(filePath)}: {bulkResponse.ElasticsearchServerError}");
        else
            Console.WriteLine($"Indexed {repo}/{Path.GetFileName(filePath)} ({group.Count()} chunks)");
    }
}, repoOption);

return await cmd.InvokeAsync(args);

// -------- helpers --------
static bool TryReadText(string path, out string text)
{
    try
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length == 0) { text = ""; return false; }
        var nonText = bytes.Take(1024).Count(b => b == 0);
        if (nonText > 0) { text = ""; return false; }

        text = File.ReadAllText(path, Encoding.UTF8);
        return !string.IsNullOrWhiteSpace(text);
    }
    catch
    {
        text = "";
        return false;
    }
}

static List<string> ChunkTextTokenAware(string text, int maxTokens = 400, int overlapTokens = 240, int minChars = 40)
{
    var safe = string.Join("\n", text.Split('\n').Where(ln => ln.Length < 2000));
    if (string.IsNullOrWhiteSpace(safe) || safe.Length < minChars)
        return new();

    var tokenizer = GptEncoding.GetEncoding("cl100k_base");
    var tokens = tokenizer.Encode(safe).ToList();
    var chunks = new List<string>();

    for (int i = 0; i < tokens.Count; i += (maxTokens - overlapTokens))
    {
        var slice = tokens.Skip(i).Take(maxTokens).ToArray();
        if (slice.Length == 0) break;

        var chunkText = tokenizer.Decode(slice).Trim();
        if (chunkText.Length >= minChars)
            chunks.Add(chunkText);
    }

    return chunks;
}

static string GuessKind(string path)
{
    var name = Path.GetFileName(path).ToLowerInvariant();
    var ext = Path.GetExtension(path).ToLowerInvariant();
    if (name.Contains("openapi") || name.Contains("swagger")) return "openapi";
    if (name.Contains("asyncapi")) return "asyncapi";
    if (ext == ".proto") return "proto";
    if (ext is ".yml" or ".yaml") return "infra";
    if (ext is ".md" or ".rst") return "doc";
    return "code";
}

static string DocId(string repo, string relPath, int i)
{
    using var sha = SHA1.Create();
    var b = sha.ComputeHash(Encoding.UTF8.GetBytes($"{repo}:{relPath}:{i}"));
    return BitConverter.ToString(b).Replace("-", "").ToLowerInvariant();
}
