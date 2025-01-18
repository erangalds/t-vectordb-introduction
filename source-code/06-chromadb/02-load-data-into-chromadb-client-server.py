import os
import pandas as pd
from chromadb import HttpClient
from chromadb.config import Settings
from ollama import Client as OllamaClient

# Set environment variables for ChromaDB server
os.environ['CHROMA_SERVER_HOST'] = 'vectordb-lab-chroma-db'
os.environ['CHROMA_SERVER_PORT'] = '8000'
# Set environment variables for tenant and database (if applicable)
os.environ['CHROMA_TENANT_NAME'] = 'ucsc'
os.environ['CHROMA_DATABASE_NAME'] = 'mylibrary'


# Initialize ChromaDB client with persistent storage settings
chromadb_settings = Settings()
chroma_client = HttpClient(host='vectordb-lab-chroma-db',port=8000)

# Initialize Ollama client
ollama_client = OllamaClient(host='http://host.docker.internal:11434')

def generate_embeddings(text):
    response = ollama_client.embed(model='nomic-embed-text', input=text)
    return response['embeddings'][0]

# Load the Excel file
df = pd.read_excel('/sample-data/mylibrary/mylibrary.xlsx')

# Convert authors and tags to lists
df['Authors'] = df['Authors'].apply(lambda x: x.split(','))
df['Tags'] = df['Tags'].apply(lambda x: x.split(','))

# Generate vector embeddings for the book titles
df['vectorized_title'] = df['Title'].apply(lambda x: generate_embeddings(x))

# Check if the collection already exists and delete it if it does
collections = chroma_client.list_collections()
collection_names = [col for col in collections]

if 'mylibrary' in collection_names:
    chroma_client.delete_collection(name='mylibrary')
    print("Existing collection 'mylibrary' deleted.")

# Create a new collection
collection = chroma_client.create_collection(name='mylibrary')

# Preparing the Data
ids = []
documents = []
embeddings = []
metadatas = []
# Upsert data into the collection
for index, row in df.iterrows():
   document = {
        'ISBN': row['ISBN'],
        'Title': row['Title'],
        'Edition': row['Edition'],
        'Published Year': row['Published Year'],
        'Publisher': row['Publisher'],
        'Authors': ','.join(row['Authors']),
        'Tags': ','.join(row['Tags'])
   }
   ids.append(str(row['ISBN']))
   documents.append(row['Title'])
   metadatas.append(document)
   embeddings.append(row['vectorized_title'])

collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings,
    ids=ids
)
print("Collection 'mylibrary' created and data loaded successfully.")

# Reconnect to ChromaDB and list collections to ensure persistence
print('\n\nReconnecting to the Chromadb Server to verify data loading..\n')

# Initializing another new chroma_client
chroma_client = HttpClient(host='vectordb-lab-chroma-db',port=8000)
collections = chroma_client.list_collections()

print("\nAvailable Data Collections:")

for col in collections:
    print(f"Collection Name: {col}")

# Connect to 'mylibrary' collection and get document count
mylibrary_collection = chroma_client.get_collection(name='mylibrary')
document_count = mylibrary_collection.count()

print(f"\nDocument count in 'mylibrary' collection: {document_count}")


