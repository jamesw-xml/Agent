using Elastic.Clients.Elasticsearch;
using Elastic.Clients.Elasticsearch.Mapping;

namespace CodeAi.Shared.Search;

public static class EsFactory
{
    public static ElasticsearchClient Create(string url)
    {
        var settings = new ElasticsearchClientSettings(new Uri(url))
            .DefaultIndex("code_chunks")
            .RequestTimeout(TimeSpan.FromMinutes(2));
        return new ElasticsearchClient(settings);
    }

    public static void CreateIndexIfNotExists(ElasticsearchClient es, string indexName)
    {
        var exists = es.Indices.ExistsAsync(indexName).Result.Exists;
        es.Indices.DeleteAsync(indexName).Wait(); // Delete if exists to ensure fresh index creation
        exists = false; // Reset exists to ensure index is created fresh
        if (!exists)
        {
            var createResponse = es.Indices.CreateAsync(new Elastic.Clients.Elasticsearch.IndexManagement.CreateIndexRequest(indexName)
            {
               Settings = new Elastic.Clients.Elasticsearch.IndexManagement.IndexSettings
               {
                   RefreshInterval = "30s"
               },
               Mappings = new Elastic.Clients.Elasticsearch.Mapping.TypeMapping
               {
                   Properties = new (new Dictionary<PropertyName, IProperty>
                   {
                       {new PropertyName("repo"), new KeywordProperty() },
                       {new PropertyName("service"), new KeywordProperty() },
                       {new PropertyName("file_path"), new KeywordProperty() },
                       {new PropertyName("symbol"), new KeywordProperty() },
                       {new PropertyName("language"),  new KeywordProperty()},
                          {new PropertyName("commit"), new KeywordProperty() },
                          {new PropertyName("kind"), new KeywordProperty() },
                          {new PropertyName("links"), new KeywordProperty() },
                          {new PropertyName("text"), new TextProperty() },
                          {new PropertyName("vec"), new DenseVectorProperty() {
                            Dims = 1024,
                            ElementType = "float",
                            Index = true,
                            Similarity = "cosine"
                          } }

                   })
               }
            }
            ).Result;
            if (!createResponse.IsValidResponse)
            {
                throw new Exception($"Failed to create index {indexName}: {createResponse.ElasticsearchServerError}");
            }
        }
    }
}