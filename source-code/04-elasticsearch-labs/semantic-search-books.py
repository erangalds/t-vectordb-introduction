from elasticsearch import Elasticsearch
from ollama import Client

# Elasticsearch credentials
username = 'your_username'
password = 'your_password'

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")

# Initialize Ollama client
ollama_client = Client(host='http://host.docker.internal:11434')

def generate_embeddings(text):
    response = ollama_client.embed(model='nomic-embed-text', input=text)
    return response['embeddings'][0]

def semantic_search_by_title(query_title):
    # Generate embedding for the query title
    query_embedding = generate_embeddings(query_title)

    # Search query using vector similarity
    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vectorized_title') + 1.0",
                    "params": {
                        "query_vector": query_embedding
                    }
                }
            }
        }
    }

    response = es.search(index='mylibrary', body=query)
    hits = response['hits']['hits']

    print("Semantic Search Results:")
    for hit in hits:
        source = hit['_source']
        print(f"Title: {source['title']}, Authors: {', '.join(source['authors'])}")

if __name__ == "__main__":
    query_title = "cloud"  # Replace with your search query
    semantic_search_by_title(query_title)
