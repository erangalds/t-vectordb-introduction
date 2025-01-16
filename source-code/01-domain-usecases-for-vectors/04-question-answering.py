from transformers import pipeline

# Load the pre-trained model and tokenizer from Hugging Face
question_answerer = pipeline("question-answering")

# Define the context and the question
context = "Hugging Face is a technology company based in New York City. It develops tools for natural language processing."
question = "Where is Hugging Face based?"

# Get the answer
result = question_answerer(question=question, context=context)

# Print the answer
print(f'Context:\n{context}\n')
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
