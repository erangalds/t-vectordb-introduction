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

def knn_semantic_search_by_title(query_title):
    # Generate embedding for the query title
    query_embedding = generate_embeddings(query_title)

    # Search query using KNN
    query = {
        "knn": {
            "field": "vectorized_title",
            "query_vector": query_embedding,
            "k": 5,
            "num_candidates": 100,
            #"similarity": "cosine"
        }
    }

    response = es.search(index='mylibrary', body={"query": query})
    hits = response['hits']['hits']

    print(f"\nSearching Book Title:\n{query_title}\n\nKNN Semantic Search Results:\n")
    for hit in hits:
        source = hit['_source']
        score = hit['_score']
        print(f"Title: {source['title']}\nAuthors: {', '.join(source['authors'])}\nRelevance Score: {score}\n")

if __name__ == "__main__":
    query_title = "cloud"  # Replace with your search query
    knn_semantic_search_by_title(query_title)
