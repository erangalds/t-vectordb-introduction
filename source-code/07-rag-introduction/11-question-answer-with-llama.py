from ollama import Client


system_prompt = """
    You are a helpful assistant.
    """

user_prompt = 'Write a small article about generative ai'

messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
]

client = Client(host='http://host.docker.internal:11434')

response = client.chat(
        'llama3.2',
        messages=messages
)

print(response.message.content)

