# ChromaDB Overview
![Chromadb Concepts](https://cookbook.chromadb.dev/core/concepts/#vector-segment)

## Terminology and Components

1. **Tenants**: A logical grouping for a set of databases, designed to model a single organization or user.
2. **Databases**: A logical grouping for a set of collections, designed to model a single application or project.
3. **Collections**: Grouping mechanism for embeddings, documents, and metadata.
4. **Documents**: Chunks of text that fit within the embedding model's context window.
5. **Metadata**: Dictionary of key-value pairs associated with an embedding.
6. **Embedding Function**: Wrappers that expose a consistent interface for generating embedding vectors from documents or text queries.
7. **Distance Function**: Functions that calculate the difference between two embedding vectors.
8. **Vector Segment**: Segment or index where embeddings are stored.

## Data Storage

ChromaDB stores data in a SQLite database. The data is organized into:
- **Sysdb**: Stores information about tenants, databases, collections, and segments.
- **Metadata Segment**: Stores metadata and documents.
- **WAL (Write-Ahead Log)**: Ensures durability of data.
- **Segments**: Each collection has two segments - metadata and vector.

## Vector Search Options

ChromaDB supports several vector search options:
1. **Cosine Similarity**: Useful for text similarity.
2. **Euclidean (L2) Distance**: Useful for text similarity, more sensitive to noise than cosine.
3. **Inner Product (IP)**: Useful for recommender systems.

ChromaDB uses the HNSW (Hierarchical Navigable Small World) algorithm for indexing and searching vectors. It also uses a brute-force index to buffer embeddings in memory before they are added to the HNSW index.
