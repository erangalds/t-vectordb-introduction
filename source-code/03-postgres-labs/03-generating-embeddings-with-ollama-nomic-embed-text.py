#from ollama import embed
import ollama

client = ollama.Client(host='http://host.docker.internal:11434')

# Example text to embed
text = "The quick brown fox jumps over the lazy dog."

# Generate embeddings for the text
response = client.embed(
    model='nomic-embed-text', input=text
)
embedding_vector = response['embeddings'][0]
# Print the embeddings
print(f'Text Needs to be converted to embeddings:\n{text}\n\n')
print(f'\nNumber of dimensions in the embedding vector: {len(embedding_vector)}\n')
print(embedding_vector)

