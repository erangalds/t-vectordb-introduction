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
]

client = Client(host='http://host.docker.internal:11434')

# Starting to Chat
while True:
    user_query = input('\nUser:')

    # Loop Break Condition
    if user_query.lower() == 'quit':
        print(f'Thank You, have a nice day!')
        break

    user = {
        'role': 'user',
        'content': user_query
    }
    messages.append(user)
    # Invoking the chat 
    response = client.chat(
        'llama3.2',
        messages=messages
    )
    # Assistant Response
    assistant_response = response.message.content
    assistant = {
        'role': 'assistant',
        'content': assistant_response 
    }
    # Adding the assistant response to the messages list
    messages.append(assistant)

    print(f'\nLLM:\n{assistant_response}\n\nEnter "Quit" to Exit')

