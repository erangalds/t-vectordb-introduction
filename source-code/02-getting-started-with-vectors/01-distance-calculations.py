import spacy
import numpy as np
from scipy.spatial.distance import cosine, euclidean

# Load the pre-trained word embeddings model
# We need to download the spacy model first as below. python -m spacy download en_core_web_md
# python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')
# Define the texts
text_a = "The cat is playing with a toy."
text_b = "A kitten is interacting with a plaything."
text_c = "The chef is cooking a delicious meal."
text_d = "Economics is the social science that studies the production, distribution, and consumption of goods and services."
text_e = "Economics studies goods and services."

# Convert the texts to vector representations using the spaCy model
vector_a = nlp(text_a).vector
vector_b = nlp(text_b).vector
vector_c = nlp(text_c).vector
vector_d = nlp(text_d).vector
vector_e = nlp(text_e).vector

# Displaying the Text and Their Vector Representations
print(f"Text : {text_a}\nVector Length: {len(vector_a)} \nVector[0-25]: {vector_a[:25]}\n")
print(f"Text : {text_b}\nVector Length: {len(vector_b)} \nVector[0-25]: {vector_b[:25]}\n")
print(f"Text : {text_c}\nVector Length: {len(vector_c)} \nVector[0-25]: {vector_c[:25]}\n")
print(f"Text : {text_d}\nVector Length: {len(vector_d)} \nVector[0-25]: {vector_d[:25]}\n")
print(f"Text : {text_e}\nVector Length: {len(vector_e)} \nVector[0-25]: {vector_e[:25]}\n")

# Calculate the cosine similarity between the vectors
# The reason we calculate 1 - cosine() to get the cosine similarity is because the cosine() function from the scipy.
# spatial.distance module computes the cosine distance rather than the cosine similarity.
# Cosine Distance measures the angle between two vectors and is defined as:
# cosine_distance = 1 − cos(𝜃)
# Whereas, Cosine Similarity measures how similar two vectors are and is defined as: 
# cosine_similarity = cos(𝜃)
# Since cosine() returns a value between 0 and 2 (where 0 means identical and 2 means completely different), 
# calculating 1 - cosine() converts it into a similarity measure that ranges between -1 and 1 (where 1 means identical and -1 means completely different).

cosine_sim_ab = 1 - cosine(vector_a, vector_b)
cosine_sim_ac = 1 - cosine(vector_a, vector_c)
cosine_sim_de = 1 - cosine(vector_d, vector_e)
print(f'Text A : {text_a}\nText B: {text_b}')
print(f"Cosine similarity between Text A and Text B: {cosine_sim_ab:.2f}\n")

print(f'Text A : {text_a}\nText C: {text_c}')
print(f"Cosine similarity between Text A and Text C: {cosine_sim_ac:.2f}\n")

print(f'Text D : {text_d}\nText E: {text_e}')
print(f"Cosine similarity between Text D and Text E: {cosine_sim_de:.2f}\n")

# Calculate the Euclidean distance between the vectors
print(f'Text A : {text_a}\nText B: {text_b}')
euclidean_dist_ab = euclidean(vector_a, vector_b)
print(f"Euclidean distance between Text A and Text B: {euclidean_dist_ab:.2f}\n")

print(f'Text A : {text_a}\nText C: {text_c}')
euclidean_dist_ac = euclidean(vector_a, vector_c)
print(f"Euclidean distance between Text A and Text C: {euclidean_dist_ac:.2f}\n")

print(f'Text D : {text_d}\nText E: {text_e}')
euclidean_dist_de = euclidean(vector_d, vector_e)
print(f"Euclidean distance between Text D and Text E: {euclidean_dist_de:.2f}\n")

# Calculate the magnitudes of the vectors
magnitude_d = np.linalg.norm(vector_d)
magnitude_e = np.linalg.norm(vector_e)

print(f'Text D : {text_d}\n')
print(f"Magnitude of Text D's vector: {magnitude_d:.2f}")

print(f'Text E : {text_e}\n')
print(f"Magnitude of Text E's vector: {magnitude_e:.2f}")



