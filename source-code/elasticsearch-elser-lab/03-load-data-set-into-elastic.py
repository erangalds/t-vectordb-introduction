from elasticsearch import Elasticsearch
import time 
import eland 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

# Loading the data set
index="elser-olympic-games"
csv_file="/sample-data/Athletes_summer_games.csv"


eland.csv_to_eland(
   csv_file,
   es_client=es,
   es_dest_index=index,
   es_if_exists='replace',
   es_dropna=True,
   es_refresh=True,
   index_col=0,
   es_type_overrides={
       "City": "text",
       "Event": "text",
       "Games": "text",
       "Medal": "text",
       "NOC": "text",
       "Name": "text",
       "Season": "text",
       "Sport": "text",
       "Team": "text"
   }
)


# Verify data load
df = eland.DataFrame(
    es_client=es,
    es_index_pattern='elser-olympic-games'
)

print(f'\n\nFirst Few Lines of the loaded dataset:\n{df.head()}')