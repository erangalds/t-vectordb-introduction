from elasticsearch import Elasticsearch

# Elasticsearch credentials
username = 'your_username'
password = 'your_password'

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")

# Define the index name
index_name = 'mylibrary'

# Define the index mapping
index_mapping = {
    "mappings": {
        "properties": {
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

def create_index():
    # Check if the index already exists
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Deleting it.")
        es.indices.delete(index=index_name)
        
    # Create the index with the specified mapping
    es.indices.create(index=index_name, body=index_mapping)
    print(f"Index '{index_name}' created successfully.")

if __name__ == "__main__":
    create_index()


