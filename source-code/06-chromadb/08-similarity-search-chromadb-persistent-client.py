import os
import pandas as pd
from chromadb import PersistentClient
from chromadb.config import Settings
from ollama import Client as OllamaClient

# Set environment variables for tenant and database (if applicable)
os.environ['CHROMA_TENANT_NAME'] = 'ucsc'
os.environ['CHROMA_DATABASE_NAME'] = 'mylibrary-persistent-local-client'

# Initialize ChromaDB client with persistent storage settings
chromadb_settings = Settings()
chroma_client = PersistentClient(path='/chromadb/')

# Initialize Ollama client
ollama_client = OllamaClient(host='http://host.docker.internal:11434')

def generate_embeddings(text):
    response = ollama_client.embed(model='nomic-embed-text', input=text)
    return response['embeddings'][0]

search_title = 'cloud'
search_title_embedding = generate_embeddings(search_title)

collection = chroma_client.get_collection(name='mylibrary-persistent-local-client')

results = collection.query(
    query_embeddings=[search_title_embedding],
    n_results=5,
)

print(f'Search Results: \n{results}')

result_ids = results['ids'][0]
result_documents = results['documents'][0]
result_distances = results['distances'][0]

print(f'\n\nSearching Book Title:\n{search_title}\n')

for i in range(len(result_ids)):
    print(f'Title : {result_documents[i]}')
    print(f'Similarity Distance: {result_distances[i]}')

