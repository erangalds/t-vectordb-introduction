import os
import chromadb
from chromadb.config import Settings

# Set environment variables for ChromaDB server host and port
os.environ['CHROMA_SERVER_HOST'] = 'vectordb-lab-chroma-db'
os.environ['CHROMA_SERVER_PORT'] = '8000'

# Create a ChromaDB client with custom settings
settings = Settings()
chroma_client = chromadb.HttpClient(host='vectordb-lab-chroma-db',port=8000)

def list_collections():
    collections = chroma_client.list_collections()
    print("Available Data Collections:")
    for collection in collections:
        print(f"Collection Name: {collection}")

if __name__ == "__main__":
    list_collections()
