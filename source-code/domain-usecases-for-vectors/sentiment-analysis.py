from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Example sentences
sentences = [
    "I'm thrilled with the new smartphone I bought; it's amazing!",
    "I had the worst experience at the restaurant last night.",
    "The movie was okay, but it didn't live up to the hype.",
    "My day was fantastic! Everything went just as planned.",
    "I'm disappointed with the service I received at the store.",
]

# Perform sentiment analysis
results = sentiment_analysis(sentences)

# Print the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}\nSentiment: {result['label']}, Score: {result['score']:.4f}\n")
