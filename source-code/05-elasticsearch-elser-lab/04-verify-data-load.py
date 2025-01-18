from elasticsearch import Elasticsearch
import time 
import eland 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

# Verify data load
df = eland.DataFrame(
    es_client=es,
    es_index_pattern='elser-olympic-games'
)

print(df.head())

# Define the index name
index_name = 'elser-olympic-games'

# Search the index to get the first 3 records
response = es.search(
    index=index_name,
    body={
        "query": {
            "match_all": {}
        },
        "_source": ["Event","Year","concatenated_text","concatenated_text_embedding"], 
        "size": 3
    }
)

# Print the values of the first 3 records
for i, hit in enumerate(response['hits']['hits']):
    print(f"Record {i+1}: {hit['_source']}")


