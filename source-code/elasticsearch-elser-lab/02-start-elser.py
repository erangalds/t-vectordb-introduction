from elasticsearch import Elasticsearch
import time 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

# A function to check the model's routing state
# https://www.elastic.co/guide/en/elasticsearch/reference/current/get-trained-models-stats.html
def get_model_routing_state(model_id=".elser_model_2"):
   try:
       status = es.ml.get_trained_models_stats(
           model_id=".elser_model_2",
       )
       return status["trained_model_stats"][0]["deployment_stats"]["nodes"][0]["routing_state"]["routing_state"]
   except:
       return None


# If ELSER is already started, then we are fine.
if get_model_routing_state(".elser_model_2") == "started":
   print("ELSER Model has been already deployed and is currently started.")


# Otherwise, we will deploy it, and monitor the routing state to make sure it is started.
else:
   print("ELSER Model will be deployed.")
   # Start trained model deployment
   es.ml.start_trained_model_deployment(
       model_id=".elser_model_2",
       #number_of_allocations=16,
       #threads_per_allocation=4,
       wait_for="starting"
   )


   while True:
       if get_model_routing_state(".elser_model_2") == "started":
           print("ELSER Model has been successfully deployed.")
           break
       else:
           print("ELSER Model is currently being deployed.")
       time.sleep(10)
