from elasticsearch import Elasticsearch
import time 
import eland

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

index="elser-olympic-games"

mappings_properties={
   "concatenated_text": {
       "type": "text"
   },
   "concatenated_text_embedding": {
       "type": "sparse_vector"
   }
}


es.indices.put_mapping(
   index=index,
   properties=mappings_properties
)

# Populate the sarse vector field
es.update_by_query(
   index="elser-olympic-games",
   pipeline="elser-ingest-pipeline",
   wait_for_completion=False
)


# Verify data load
df = eland.DataFrame(
    es_client=es,
    es_index_pattern='elser-olympic-games'
)

print(df.head())
