import pandas as pd
from elasticsearch import Elasticsearch, helpers
from ollama import Client

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")

# Initialize Ollama client
ollama_client = Client(host='http://host.docker.internal:11434')

def generate_embeddings(text):
    response = ollama_client.embed(model='nomic-embed-text', input=text)
    return response['embeddings'][0]

def read_excel_file(filepath, sheet_name):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def create_or_recreate_index(): # Check if index exists 
    if es.indices.exists(index='mylibrary'): 
        print("Index 'mylibrary' already exists. Deleting it.") 
        es.indices.delete(index='mylibrary') 
    # Create index with the new mapping 
    mapping = { 
        "mappings": 
            { 
                "properties": 
                    { 
                        "isbn": {"type": "text"}, 
                        "title": {"type": "text", "analyzer": "standard"}, 
                        "edition": {"type": "text"}, 
                        "published_year": {"type": "integer"}, 
                        "publisher": {"type": "text"}, 
                        "authors": {"type": "text"}, 
                        "tags": {"type": "text"}, 
                        "vectorized_title": {"type": "dense_vector", "dims": 768} 
                    } 
            } 
    } 
    es.indices.create(index='mylibrary', body=mapping)

def bulk_insert_books(df):
    actions = [
        {
            "_index": "mylibrary",
            "_id": i,
            "_source": {
                "isbn": row["ISBN"],
                "title": row["Title"],
                "edition": row["Edition"],
                "published_year": row["Published Year"],
                "publisher": row["Publisher"],
                "authors": row["Authors"].split(','),
                "tags": row["Tags"].split(','),
                "vectorized_title": generate_embeddings(row["Title"])
            }
        }
        for i, row in df.iterrows()
    ]

    #print(f'Showing the first Record:\n{actions[0]}\n\n')
    # Perform bulk insert and capture errors 
    success, failed = helpers.bulk(es, actions, raise_on_error=False, stats_only=False) 
    print(f"Successfully indexed {success} documents.") 
    if failed: 
        print(f"Failed to index {len(failed)} documents.")
        for failure in failed:
            print(f'Failure: {failure}')

if __name__ == "__main__":
    # Create or Recreate the index
    create_or_recreate_index()
    # Read the Excel File
    df = read_excel_file('/sample-data/mylibrary/mylibrary.xlsx', 'Sheet1')
    # Inserting Data into the Elasticsearch Index
    bulk_insert_books(df)

    
