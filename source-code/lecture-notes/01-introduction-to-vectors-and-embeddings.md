# Introduction to Vectors and Embeddings

## Roles of *Supervised and Unsupervised Learning* in *Vector Search*
In *Supervised Learning*, a *model learns* from *labeled data* to determine a *connection or mapping* between the *input features and output labels*. While the *model development phase also called training phase*, the *model* adjusts its *parameters* in such a way which will *minimize the error* between its *prediction* and the *true labels*. This *supervised learning ML* technique is frequently used in ***Natural Language Processing (NLP)***. The challenge with this *ML technique* is getting the *labeled data set* prepared for the *model development*. 

On the other hand we have another *ML technique* called *Unsupervised Learning*, where the *model discovers patterns and structures* in the *input data* without using *labeled examples*. This *ML technique* is widely used on *finding similarities, differences or underlying structures* within data. The different types of techniques used under *supervised learning* are *clustering, dimensionality reduction and density estimation*. This *ML technique* is widely used in *anomaly detection, data compression and feature extraction*. 

When we talk to about *vector search and NLP*, out of the above two, *supervised learning* can be used to generate *word or sentence embeddings*. This is done by a *model* to predict the *context* of a given word based on its *neighboring words* or else we can use a *model* to *classify documents* into predefined categories. By doing this, we can capture *semantic and syntactic relationships* in the text. 

On the other hand *unsupervised learning* can be employed to generate *embeddings* by identifying patterns and similarities in textual data. *word2vec and Glo Ve* are such models. 

## What is an *embedding also called a vector*?
Embedding is a *numerical* representation of a *complex data*. In here, the *complex data item* could be a *word, phrase, document image, video or a sound*. These *vector representation* can be used for *sentiment analysis, machine translation, text classification, image recognition, object detection, and image retrieval*. 

## What challenges are *vectors* solving?
***BM25*** is a widely used text retrieval algorithm based on *probabilistic information retrieval theory*. *BM25* algorithm ranks the *documents* based on the *frequency of query terms* in the *document*. The points which it uses are *term frequency, inverse document frequency and document length normalization*. 

The challenge with *BM25* algorithm is that, it relies heavily on exact *term matches*. Sometimes this can retrieve less relevant results specially when dealing with synonyms, misspellings, or subtle semantic variations. The other drawback is *BM25* does not capture the *contextual relationship* between words, therefore it is less effective in understanding the meaning of *phrases or sentences*.

*Vector Search*, includes both exact match and ***approximate nearest neighbor (ANN)*** search is able to address some of the above mentioned limitations of *BM25* algorithm. The *vector* is able to capture the *semantic and contextual relationship* between words, phrases and documents. The similarity between the *search query and document vectors* is used to determine the relevance. 

Common use cases for *vector search*
+ E-commerce product search
+ Document retrieval
+ Question Answering systems. 
+ Image recognition and retrieval
+ Music recommendation
+ Security and User and Entity Behavior Analytics (UEBA)

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

