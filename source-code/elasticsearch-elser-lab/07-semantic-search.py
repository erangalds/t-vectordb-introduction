from elasticsearch import Elasticsearch
import time 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')

def semantic_search(search_text):
   response = es.search(
       index="elser-olympic-games",
       size=3,
       query={
           "bool": {
           "must": [
               {
                   "text_expansion": {
                       "concatenated_text_embedding": {
                       "model_id": ".elser_model_2",
                       "model_text": search_text
                       }
                   }
               },
               {
                   "exists": {
                       "field": "Medal"
                   }
               }
           ]
           }
       },
       source_excludes="*_embedding, concatenated_text"
   )


   for hit in response["hits"]["hits"]:
       doc_id = hit["_id"]
       score = hit["_score"]
       year = hit["_source"]["Year"]
       event = hit["_source"]["Event"]
       games = hit["_source"]["Games"]
       sport = hit["_source"]["Sport"]
       city = hit["_source"]["City"]
       team = hit["_source"]["Team"]
       name = hit["_source"]["Name"]
       medal = hit["_source"]["Medal"]


       print(f"Score: {score}\nDocument ID: {doc_id}\nYear: {year}\nEvent: {event}\nName: {name}\nCity: {city}\nTeam: {team}\nMedal: {medal}\n")


print('\n\nSemantic Search Example:\n\n')
search_text = "Who won the Golf competition in 1900?"
print(f'Searching for :\n{search_text}\n')
semantic_search(search_text)


search_text = "2004 Women's Marathon winners"
#print(f'Searching for :\n{search_text}\n')
#semantic_search(search_text)

search_text = "Women archery winners of 1908"
#print(f'Searching for :\n{search_text}\n')
#semantic_search(search_text)





