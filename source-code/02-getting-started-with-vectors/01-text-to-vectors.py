from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define the text string
text = "Hugging Face provides state-of-the-art NLP models."

# Tokenize the text and convert to input IDs and attention mask
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get the embeddings from the model
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

# Use the embeddings of the [CLS] token as the vector representation of the text
text_vector = last_hidden_states[:, 0, :].squeeze()

print(f"Text: {text}")
print(f"Vector: {text_vector}")
