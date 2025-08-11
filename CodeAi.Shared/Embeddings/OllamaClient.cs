using System.Net.Http.Json;
using System.Text.Json;

namespace CodeAi.Shared.Embeddings;

public sealed class OllamaClient
{
    private readonly HttpClient _http;
    private readonly string _embedModel;
    private readonly string _genModel;

    public OllamaClient(HttpClient http, string embedModel = "mxbai-embed-large", string genModel = "llama3.1:8b")
    {
        _http = http;
        _embedModel = embedModel;
        _genModel = genModel;
        _http.Timeout = TimeSpan.FromMinutes(2);
        if (_http.BaseAddress is null) _http.BaseAddress = new Uri("http://localhost:11434");
    }

    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        var req = new { model = _embedModel, prompt = text };  // correct for Ollama embeddings
        using var r = await _http.PostAsJsonAsync("/api/embeddings", req, ct);
        r.EnsureSuccessStatusCode();

        var json = await r.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);

        float[] vec;

        // mxbai-embed-large → { "embedding": [ ... ] }
        if (json.TryGetProperty("embedding", out var single) && single.ValueKind == JsonValueKind.Array)
        {
            vec = single.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        }
        // some models → { "embeddings": [[ ... ], [ ... ], ...] }
        else if (json.TryGetProperty("embeddings", out var multi) &&
                 multi.ValueKind == JsonValueKind.Array &&
                 multi.GetArrayLength() > 0 &&
                 multi[0].ValueKind == JsonValueKind.Array)
        {
            vec = multi[0].EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        }
        else
        {
            throw new InvalidOperationException("Ollama embeddings response missing 'embedding(s)'.");
        }

        if (vec.Length == 0) throw new InvalidOperationException("Received empty embedding vector.");
        if (vec.Length != 1024) throw new InvalidOperationException($"Embedding dims {vec.Length} != 1024 (mxbai-embed-large).");

        return vec;
    }

    public async Task<string> GenerateAsync(string prompt, CancellationToken ct = default)
    {
        var req = new { model = _genModel, prompt, stream = false };
        using var r = await _http.PostAsJsonAsync("/api/generate", req, ct);
        r.EnsureSuccessStatusCode();
        var json = await r.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
        return json.GetProperty("response").GetString() ?? "";
    }
}