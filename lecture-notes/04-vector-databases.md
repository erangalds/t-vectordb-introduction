# Vector Databases
## What is a Vector Database ?
A **vector database** is a specialized database designed to store and manage data as high-dimensional vectors. Unlike traditional databases that handle structured data in rows and columns, vector databases store data as vectors, which are mathematical representations of features or attributes contained in data. Each vector consists of a specific number of dimensions, which can vary from just a few dozen to several thousand, depending on the complexity and granularity of the data.

## Benefits of Vector Databases:
1. **Efficient Similarity Searches**: Vector databases excel at performing similarity-based searches, making them ideal for applications involving natural language processing (NLP), recommendation systems, and image retrieval.
2. **Handling Unstructured Data**: They are well-suited for managing unstructured data, such as text, images, and audio, which are increasingly prevalent in today's data landscape.
3. **Scalability**: Vector databases can efficiently handle large volumes of high-dimensional data, making them scalable for various AI and machine learning applications.
4. **Enhanced Search Capabilities**: They enable hybrid search, combining traditional keyword-based search with semantic similarity search, providing more relevant and accurate results.

## Difference Between Traditional Relational or NoSQL Databases and Vector Databases

### Traditional Relational Databases:
- **Data Model**: Organize data in tables with rows and columns.
- **Query Language**: Use Structured Query Language (SQL) for data manipulation.
- **Data Handling**: Best suited for structured data with a well-defined schema.
- **Use Cases**: Transactional systems, data warehousing, and applications requiring ACID properties (Atomicity, Consistency, Isolation, Durability).

### NoSQL Databases:
- **Data Model**: Can handle unstructured and semi-structured data, with flexible schemas.
- **Query Language**: Varies by type (e.g., key-value, document, column-family, graph).
- **Data Handling**: Suitable for large-scale data storage and retrieval, often used in big data and real-time applications.
- **Use Cases**: Content management systems, real-time analytics, and applications requiring high write throughput.

### Vector Databases:
- **Data Model**: Store data as high-dimensional vectors.
- **Query Language**: Specialized algorithms for similarity searches and nearest-neighbor queries.
- **Data Handling**: Ideal for unstructured data and complex, multi-dimensional data.
- **Use Cases**: AI and machine learning applications, recommendation systems, and semantic search.


## Using PostgreSQL as a Vector Database

PostgreSQL can be used as a vector database with the help of extensions like **pgvector**. This extension adds support for vector operations and similarity searches, allowing you to store, index, and query vector data directly within your PostgreSQL database.

### Main Vector Search Algorithms Supported in PostgreSQL:

#### Exact Nearest Neighbor Search (Brute Force Nearset Neighbor Search):
   - **Description**: Finds the exact nearest neighbors based on Euclidean distance, cosine similarity, inner product and manhaten distance. This method is highly accurate but can be computationally expensive for large datasets.

When we install and configure the `vector extension` the default search algorithm is *Exact Nearest Neighbor*. Below command can be used to create the `vector extension`. 

```sql
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA pg_catalog;
```

Once this extension is created the a full table scan will be used (brute force, comparing all records) for the vector search.

   - **When to Use**: Use this algorithm when precision is critical, such as in medical applications or when dealing with small to moderately sized datasets.
   - **Example Use Case**: Identifying similar medical images for diagnostic purposes.

#### Approximate Nearest Neighbor Search:
   - **Description**: Uses algorithms like Hierarchical Navigable Small World (HNSW) and Inverted File Index (IVFFLAT) to perform faster searches with a trade-off in accuracy. These algorithms provide a balance between speed and accuracy.

After creating the `vector extension` on top of that we should use `vector indexes` (vector search optimizing algorithms) to enable Approximate Nearest Neighbor Search. The two `vector indexing strategies` supported by `pgvector` is `ivfflat and hnsw`. 

   - **When to Use**: Use this algorithm when you need faster search times and can tolerate slight inaccuracies. Suitable for large-scale applications where real-time response is necessary.
   - **Example Use Case**: Real-time recommendation systems and large-scale image retrieval.

##### Hierarchical Navigable Small World (HNSW):
   - **Description**: A graph-based algorithm that constructs a multi-layered graph for efficient nearest neighbor search.
   - **Advantages**: Offers fast search times with high accuracy.
   - **When to Use**: Best for scenarios where you need quick retrieval and can afford some initial setup time for building the graph.
   - **Example Use Case**: Large-scale search applications like e-commerce product recommendations.

We can create HNSW Vector Indexes (Vector Search Optimization Algorithm) as below. 

```sql
-- First Create the Vector Extension
CREATE EXTENSION IF NOT EXISTS vector;
-- Creating the Data Table
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(3)
);
-- Creating the HNSW Index
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```


##### Inverted File Index (IVF):
   - **Description**: A method that partitions the vector space into clusters and indexes them for efficient search.
   - **Advantages**: Provides a good trade-off between search speed and accuracy.
   - **When to Use**: Useful for applications where the dataset is large, and search speed is a priority.
   - **Example Use Case**: Searching large multimedia databases.

We can create IVFFLAT Vector Indexes (Vector Search Optimization Algorithm) as below. 

```sql
-- First Create the Vector Extension
CREATE EXTENSION IF NOT EXISTS vector;
-- Creating the Data Table
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(3)
);
-- Creating the HNSW Index
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

[IVFFLAT vs HNSW Comparison](https://medium.com/@bavalpreetsinghh/pgvector-hnsw-vs-ivfflat-a-comprehensive-study-21ce0aaab931)

## Using Elasticsearch as a Vector Database

Elasticsearch can also be used as a vector database with the help of plugins like **Elasticsearch Vector Search (Elasticsearch Vectors)**. This plugin allows you to store and query vector data efficiently.

### Main Vector Search Algorithms Supported in Elasticsearch:

1. **k-Nearest Neighbors (k-NN)**:
   - **Description**: Finds the k closest vectors based on cosine similarity or Euclidean distance. This algorithm is straightforward and effective for small to medium-sized datasets.
   - **When to Use**: Use k-NN for applications where you need high accuracy and can handle the computational cost. Suitable for datasets where the search space is not extremely large.
   - **Example Use Case**: Content-based image retrieval in a photo-sharing application.

2. **Approximate Nearest Neighbor (ANN)**:
   - **Description**: Uses algorithms like HNSW to perform faster searches with a trade-off in accuracy. ANN algorithms are optimized for large-scale datasets.
   - **When to Use**: Use ANN for applications where speed is critical, and you can tolerate slight inaccuracies. Ideal for real-time applications with large datasets.
   - **Example Use Case**: Real-time recommendations in a streaming service.

### Hierarchical Navigable Small World (HNSW) in Elasticsearch:
   - **Description**: A graph-based algorithm integrated into Elasticsearch for efficient nearest neighbor search.
   - **Advantages**: Fast and scalable with high search accuracy.
   - **When to Use**: Best for applications requiring quick retrieval from large datasets with an acceptable setup time.
   - **Example Use Case**: Large-scale recommendation systems and personalized content delivery.

## Using ChromaDB as a Vector Database
Vector searching is a powerful technique used in machine learning and information retrieval to find the closest matches to a given vector within a dataset. ChromaDB is a state-of-the-art database specifically optimized for handling and querying vector data efficiently. It supports various vector operations, including similarity searches, which are crucial for applications like image recognition, recommendation systems, and natural language processing.

Different ways (Modes) which we can run ChromaDB
+ Ephemeral Client - Here, the ChromaDB will run in Memory on the local machine
+ Persistant Client - Here, the ChromaDB will run in Memory with Data Persistancy on local machine. 
+ HTTPClient - Here, the ChromaDB will run in Memory with Data Persistancy on a Remote machine.

By understanding and leveraging these algorithms and their use cases, you can effectively implement vector search capabilities in PostgreSQL and Elasticsearch and ChromaDB, enhancing your ability to handle high-dimensional data efficiently.
