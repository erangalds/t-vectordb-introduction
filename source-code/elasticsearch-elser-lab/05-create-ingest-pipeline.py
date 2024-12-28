from elasticsearch import Elasticsearch
import time 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

es.ingest.put_pipeline(
   id="elser-ingest-pipeline",
   description="Ingest pipeline for ELSER",
   processors=[
       {
           "script": {
           "description": "Concatenate some selected fields value into `concatenated_text` field",
           "lang": "painless",
           "source": """
               ctx['concatenated_text'] = ctx['Name'] + ' ' + ctx['Team'] + ' ' + ctx['Games'] + ' ' + ctx['City'] + ' ' + ctx['Event'];
           """
           }
       },
       {
           "inference": {
               "model_id": ".elser_model_2",
               "ignore_missing": True,
               "input_output": [
                   {
                       "input_field": "concatenated_text",
                       "output_field": "concatenated_text_embedding"
                   }
               ]
           }
       }
   ]
)

pipelines = es.ingest.get_pipeline()

for pipeline_id in pipelines:
    print(f'Pipeline ID: {pipeline_id}')

pipeline_config = es.ingest.get_pipeline(id='elser-ingest-pipeline')

print(pipeline_config['elser-ingest-pipeline'])