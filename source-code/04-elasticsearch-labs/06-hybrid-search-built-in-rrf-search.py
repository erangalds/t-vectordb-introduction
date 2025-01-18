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

def hybrid_search_by_title(query_title):
    # Generate embedding for the query title
    query_embedding = generate_embeddings(query_title)

    # Search query using KNN
    standard_search_query = {
        "query": {
            "match": {
                "title": query_title
            }
        }
    }
    semantic_search_query = {
        "knn": {
            "field": "vectorized_title",
            "query_vector": query_embedding,
            "k": 5,
            "num_candidates": 100,
            #"similarity": "cosine"
        }
    }

    combined_query = {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard" : standard_search_query,
                    },
                    semantic_search_query
                ],
                "rank_window_size": 50,
                "rank_constant":20
            }
        }
    }

    response = es.search(index='mylibrary', body=combined_query)
    hits = response['hits']['hits']

    print(f"Searching Book Title:\n{query_title}\n\nHybrid Search Results:")
    for hit in hits:
        source = hit['_source']
        score = hit['_score']
        print(f"Title: {source['title']}\nAuthors: {', '.join(source['authors'])}\nRelevance Score: {score}\n")

if __name__ == "__main__":
    query_title = "cloud"  # Replace with your search query
    hybrid_search_by_title(query_title)
