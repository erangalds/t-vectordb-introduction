from ollama import Client

# Replace 'http://your-host-url:port' with the actual URL of your hosted service
client = Client(host='http://host.docker.internal:11434')

# Now you can use the client to make requests
response = client.chat(
    model='llama3.2', 
    messages=[
        {
            'role': 'user', 
            'content': 'Why is the sky blue?'
        }
    ]
)
print(response['message']['content'])
