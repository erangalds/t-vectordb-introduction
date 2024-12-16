#from ollama import embed
import ollama

client = ollama.Client(host='http://host.docker.internal:11434')

# Example text to embed
text = "The quick brown fox jumps over the lazy dog."

# Generate embeddings for the text
response = client.embed(
    model='nomic-embed-text', input=text
)

# Print the embeddings
print(response['embeddings'])

