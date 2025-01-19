from ollama import Client

# Multi Query: Different Perspectives

system_prompt = """You are a helpful assistant that generates 3 sub-questions related to a user input question.
The goal is to break down the user input into a set of sub-problems / sub-questions that can be answers in isolation.

Output Format:
Output should only contain the different versions of the qustions.
Give the five different versions as a python list
"""

# system_prompt = """You are an AI language model assistant. Your task is to generate five 
# different versions of the given user question to retrieve relevant documents from a vector 
# database. By generating multiple perspectives on the user question, your goal is to help
# the user overcome some of the limitations of the distance-based similarity search. 
# Provide these alternative questions separated by newlines.

# Output Format:
# Output should only contain the different versions of the qustions.
# Give the five different versions as a python list
# """
user_prompt = 'what is the planed solar capacity increase for next few years in Sri Lanka?'

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






