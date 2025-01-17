# Introduction to Vectors and Embeddings
https://www.timescale.com/blog/a-beginners-guide-to-vector-embeddings

## Roles of *Supervised and Unsupervised Learning* in *Vector Search*
In *Supervised Learning*, a *model learns* from *labeled data* to determine a *connection or mapping* between the *input features and output labels*. While the *model development phase also called training phase*, the *model* adjusts its *parameters* in such a way which will *minimize the error* between its *prediction* and the *true labels*. This *supervised learning ML* technique is frequently used in ***Natural Language Processing (NLP)***. The challenge with this *ML technique* is getting the *labeled data set* prepared for the *model development*. 

On the other hand we have another *ML technique* called *Unsupervised Learning*, where the *model discovers patterns and structures* in the *input data* without using *labeled examples*. This *ML technique* is widely used on *finding similarities, differences or underlying structures* within data. The different types of techniques used under *supervised learning* are *clustering, dimensionality reduction and density estimation*. This *ML technique* is widely used in *anomaly detection, data compression and feature extraction*. 

Both supervised and unsupervised learning models can generate embeddings, but they do so in slightly different ways:

Supervised Learning Models: In this approach, embeddings are often learned as part of a larger task, such as classification or regression. For example, in natural language processing (NLP), models like BERT and GPT learn to generate embeddings while being trained to predict the next word or classify text. These embeddings are highly contextual and tailored to the specific task.

Unsupervised Learning Models: Unsupervised learning models generate embeddings without any labeled data. Techniques like Word2Vec and FastText are examples of unsupervised methods that create embeddings by analyzing word co-occurrence patterns in a large corpus of text. These embeddings capture semantic relationships between words based on their usage in different contexts.

Supervised Learning Model Example: BERT
BERT (Bidirectional Encoder Representations from Transformers) is a popular model for generating embeddings in a supervised manner.

Training Objective: BERT is trained on two tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). In MLM, certain words in a sentence are masked, and the model learns to predict them. In NSP, the model predicts whether two sentences follow each other in the text.

Embeddings Generation: During this training, BERT learns to generate embeddings for each word in a sentence. These embeddings capture the context in which words appear. For instance, the word "bank" would have different embeddings based on whether it's used in a financial context or a river context.

Example: If we input the sentence "The cat sat on the mat," BERT will produce embeddings for each word that reflect their contextual meanings. These embeddings are used for downstream tasks like classification, where each word's vector represents its meaning in context.

Unsupervised Learning Model Example: Word2Vec
Word2Vec is an unsupervised model that generates word embeddings based on word co-occurrence patterns in large text corpora.

Training Objective: Word2Vec uses two techniques—Continuous Bag of Words (CBOW) and Skip-gram. In CBOW, the model predicts a word based on its surrounding context words. In Skip-gram, it does the reverse: predicts surrounding words given a target word.

Embeddings Generation: Through these methods, Word2Vec learns vector representations of words that capture their semantic relationships. Words used in similar contexts have similar embeddings.

Example: If we train Word2Vec on a large corpus of text, and input the word "king," it might generate an embedding close to the embeddings for "queen," "prince," and "monarch." This shows that the model understands the semantic relationships between these words.

## What is an *embedding also called a vector*?
Embedding is a *numerical* representation of a *complex data*. In here, the *complex data item* could be a *word, phrase, document image, video or a sound*. These *vector representation* can be used for *sentiment analysis, machine translation, text classification, image recognition, object detection, and image retrieval*. 

## What challenges are *vectors* solving?
***BM25*** is a widely used text retrieval algorithm based on *probabilistic information retrieval theory*. *BM25* algorithm ranks the *documents* based on the *frequency of query terms* in the *document*. The points which it uses are *term frequency, inverse document frequency and document length normalization*. 

The challenge with *BM25* algorithm is that, it relies heavily on exact *term matches*. Sometimes this can retrieve less relevant results specially when dealing with synonyms, misspellings, or subtle semantic variations. The other drawback is *BM25* does not capture the *contextual relationship* between words, therefore it is less effective in understanding the meaning of *phrases or sentences*.

*Vector Search*, includes both exact match and ***approximate nearest neighbor (ANN)*** search is able to address some of the above mentioned limitations of *BM25* algorithm. The *vector* is able to capture the *semantic and contextual relationship* between words, phrases and documents. The similarity between the *search query and document vectors* is used to determine the relevance. 

## Used cases and Domains of applications for Vectors and Vector Search
### Named Entity Recognition (NER)
***NER*** is a component of *NLP* that detects and classifies *named entities* within *unstructured text*. Here the *entity* could either be *names of persons or animals, brands, locations or even organizations*. *Supervised learning technique* is heavily used in this. 

One of the use cases of *NER* is handling ***Personally Identifiable Information (PII)***. Financial services companies user *NER* to mitigate the risk of PII exposure, by employing it on both the *stale data and inflight data*. Once identified, we can *redact, encrypt, mask or set access controls* to that data set. 

### Sentiment Analysis
*Sentiment Analysis* is another technique used to identify and extract the *sentiment* of a *unstructured text*. A lot of companies now use this to understand the customer sentiments about their products. 

### Text Classification
*Text Classification* assigns predefined set of categories to unstructured data. A real-life example would be email spam management. 

### Question Answering (QA)
QA is a field in NLP, which is to build systems which automatically answers questions posed by users. Its a very challenging requirement, mainly because we have to identify the *context* within the *scope* of the question. Sometimes when the question is a part of a conversation, and multiple conversations can be interwind in a given session. 

### Text Summarization
*Text Summarization* is another challenging NLP Task. This is due to the requirement of understanding of the *semantics* of the *document* and identifying *most relevant information* to keep. Because, while reducing the text we have to make sure to keep the *meaning and accuracy*. 

