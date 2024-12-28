from elasticsearch import Elasticsearch
import time 

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")
# Displaying  the connection to elastic  cluster
print(f'Elasticserch Connection Info:\n{es.info()}')


# Get a list of currently available trained ml models
# Get the list of trained models 
response = es.cat.ml_trained_models(format='json')

# print(response)
# Print the list of trained models
print(f'\nPrinting Available Trained Models\n')

for model in response:
    print(f'Model ID: {model['id']} ')

# Using another method to get the list of available models
try:
    # Using another method with es.ml module to get the list of available trained models.
    response = es.ml.get_trained_models()
    #print(response)
    trained_model_configs = response['trained_model_configs']
    #print(trained_model_config)
    for model in trained_model_configs:
        print(f'Model ID: {model['model_id']}')
except Exception as e:
    print(e)


# Deleting the elsermodel if deployed previously
try:
    es.ml.delete_trained_model(
        model_id='.elser_model_2',force=True
    )
    print('Model Deleted successfully. We will load it as a fresh model again')
except Exception as e:
    print('Error Occurred')
    print(e)

# Create the ELSER model configuration. Automatically download the model if it doesn't exist
try:
    es.ml.put_trained_model(
        model_id='.elser_model_2',
        input={
            'field_names': [
                'concatenated_textl'
            ]
        }
    )

    # Check the download and deploy progress
    while True:
        status = es.ml.get_trained_models(
            model_id='.elser_model_2',include='definition_status'
        )

        if status['trained_model_configs'][0]['fully_defined']:
            print(f'ELSER Model is downloaded and ready to be deployed.')
            break
        else:
            print('ELSER model is downloaded but not ready to be deployed')
        time.sleep(5)

except Exception as e:
    print('Error Occurred:')
    print(e)
