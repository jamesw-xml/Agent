namespace CodeAi.Shared.Models;

public sealed class ChunkDocument
{
    public string Repo { get; set; } = "";
    public string Service { get; set; } = "";
    public string FilePath { get; set; } = "";
    public string? Symbol { get; set; }
    public string Language { get; set; } = "";
    public string Commit { get; set; } = "";
    public string Kind { get; set; } = "";   // code|doc|openapi|infra|proto
    public string Text { get; set; } = "";
    public float[] Vec { get; set; } = Array.Empty<float>();
    public string[] Links { get; set; } = Array.Empty<string>();
}