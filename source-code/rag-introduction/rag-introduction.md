# Retrieval Augmented Generation (RAG)

## Key Supporting Topics
+ Document Loaders
+ Text Splitters
+ Retrievers

### Document Loaders
Document Loaders load data into the standard Langchain Document format. Each Document Loader has its own specific parameters, but they can all be invoked in the same way. Below is an example of using the PyPDFLoader from langchain which uses pypdf package. 

```python
from langchain_community.document_loaders import PyPDFLoader
# Document Loaders
# Document Loaders load data into the standard Langchain Document format
# Each Document Loader has its own specific parameters, but they can all be invoked in the same way. 
# Below is an example of using the PyPDFLoader from langchain which uses pypdf package.

loader = PyPDFLoader(
    "/sample-data/energy-policy.pdf",
)

docs = loader.load()

print(f'Number of Documents : {len(docs)}')

for doc in docs:
    print(f'\n Document:==========')
    print(f'\nMeta Data:\n{doc.metadata}\n')
    print(f'\nPage Content:\n{doc.page_content}')
```

### Text Splitters
Splitting documents is a critical preprocessing step for many applications. It involves dividing extensive texts into smaller, more manageable segments. This technique provides several advantages, including consistent handling of documents of different lengths, bypassing input size limitations of models, and enhancing the quality of text representations used in retrieval systems. There are various methods for splitting documents, each offering unique benefits.

   Single Document
          |
          v
   Splitting Process
          |
    ---------------------
    |   |   |   |   |   |
   v    v   v   v   v   v
 Chunk1 Chunk2 Chunk3 Chunk4 Chunk5

### Benefits of splitting documents. 
Reasons to Split Documents:
+ Managing Varied Document Lengths: In real-world collections, documents often vary in size. Splitting them ensures consistent processing across all documents.

+ Addressing Model Limitations: Many embedding and language models have constraints on maximum input size. Splitting allows for processing of documents that would otherwise exceed these limits.

+ Enhancing Representation Quality: For lengthy documents, the quality of embeddings or representations can diminish as they try to encompass too much information. Splitting results in more focused and precise representations for each segment.

+ Improving Retrieval Accuracy: In information retrieval systems, splitting can enhance the granularity of search results, enabling more precise alignment of queries with relevant document sections.

+ Optimizing Computational Efficiency: Working with smaller text chunks can be more memory-efficient and facilitate better parallel processing of tasks.

### Chunking Approaches
As of the date of writing this, there are couple of different approaches to text splitting. 
+ Length Based
+ Text Structure Based
+ Document Structure Based
+ Semantic Meaning Based

#### Legth Based
The most straightforward approach is to split documents based on their length, ensuring that each chunk remains within a specified size limit. This method is both simple and effective, offering several key benefits:

+ Easy Implementation: The process is straightforward and uncomplicated.
+ Uniform Chunk Sizes: Ensures that each chunk is of a consistent size.
+ Adaptable to Models: Easily tailored to meet the requirements of different models.

##### Types of Length-Based Splitting:
###### Token-Based: Splits text based on the number of tokens. This method is particularly useful when working with language models.

```python
from langchain.text_splitter import TokenTextSplitter

text = "Your long document text here..."

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # OpenAI's encoding
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
```

Here you can see the *document* gets *chunked* based on the *tokens* using the `encoding_name="cl100k_base"` as the *encoding scheme*, which is the *OpenAI's encoding*. The *chunks* will be broken into 100 tokens per *chunk*.

When to Use:
+ When working with models that are sensitive to token limits.
+ To guarantee that chunks stay within the model's token constraints.
+ For achieving more accurate control over input sizes for language models.

###### Character-Based: Splits text based on the number of characters, providing consistency across various types of text.

```python
from langchain.text_splitter import CharacterTextSplitter

text = "Your long document text here..."

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text) #you can also split documents using split_documents
```

Here, the `separator` is used to split the document. Example, in the above code, the `\n\n` is used as the `separator`. Therefore, `CharacterTextSplitter` function will first split the document using the `\n\n` and then calculate the `chunk_size`. If the `chunk_size` is greater than the specified figure, then it will try to further split the *chunks* to further smaller *chunks* based on the *separator*. But if unable to find the *separator* character then will stop further chunking. 

When to Use:
+ For documents with a uniform structure and formatting.
+ When a simple, character-based split is required.
+ In situations where advanced splitting methods aren't needed

#### Text Structure Based
Text is naturally structured into hierarchical units like paragraphs, sentences, and words. We can use this inherent organization to guide our splitting strategy, ensuring splits that preserve natural language flow, maintain semantic coherence, and adapt to various levels of text granularity. LangChain's RecursiveCharacterTextSplitter exemplifies this approach:

+ The RecursiveCharacterTextSplitter strives to keep larger units (e.g., paragraphs) intact.
+ If a unit exceeds the chunk size, it proceeds to the next level (e.g., sentences).
+ This process continues down to the word level, if necessary.

When to Use:
+ As the go-to option for most general-purpose text splitting tasks.
+ When handling a variety of document types with differing structures.
+ To preserve semantic coherence as much as possible in the splits.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Your long document text here..."

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = splitter.split_text(text)
```

Here the splitter will try to split on the double newlines first, then single newlines, then spaces and finally individual characters if necessary. 

#### Document Structure based
Some documents inherently possess a structured format, such as HTML, Markdown, or JSON files. Leveraging this structure for splitting is advantageous as it often groups semantically related text naturally. Key benefits of structure-based splitting include:

+ Preserving Logical Organization: Keeps the document's logical flow intact.
+ Maintaining Context: Ensures context is retained within each chunk.
+ Enhancing Effectiveness: More effective for downstream tasks like retrieval or summarization.

Examples of Structure-Based Splitting:
+ Markdown: Split based on headers (e.g., #, ##, ###).
+ HTML: Split using tags.
+ JSON: Split by object or array elements.
+ Code: Split by functions, classes, or logical blocks.

##### Markdown Splitter
The `MarkdownHeaderTextSplitter` is purpose-built to manage Markdown documents, ensuring the integrity of header hierarchy and the overall document structure.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """
# Title
## Section 1
Content of section 1
## Section 2
Content of section 2
### Subsection 2.1
Content of subsection 2.1
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
```

When to Use:
+ For Markdown Documents: Specifically designed for handling Markdown files.
+ To Preserve Logical Structure: Ideal for maintaining the hierarchical organization of documentation or articles.
+ When Header-Based Organization Matters: Essential when header-based structuring is critical for your Retrieval-Augmented Generation (RAG) application.

##### HTML Text Splitter
Dividing HTML documents into manageable chunks is crucial for numerous text processing tasks, including natural language processing, search indexing, and more. There are three ways to split HTML documents. 
+ HTMLHeaderTextSplitter
+ HTMLSectionSplitter
+ HTMLSemanticPreservingSplitter

###### HTMLHeaderTextSplitter
The splitter divides HTML text using header tags (e.g., <h1>, <h2>, <h3>, etc.), and appends relevant metadata to each chunk.

Capabilities:
+ Splits text at the level of HTML elements.
+ Retains context-rich information embedded in the document's structure.
+ Can return chunks on an element-by-element basis or combine elements that share the same metadata.

##### HTMLSectionSplitter
This splitter is similar to the HTMLHeaderTextSplitter but targets splitting HTML into sections based on specified tags.

Capabilities:
+ Utilizes XSLT transformations to identify and split sections.
+ Employs RecursiveCharacterTextSplitter for handling large sections.
+ Takes font sizes into account to determine sections.

##### HTMLSemanticPreservingSplitter
This splitter breaks HTML content into manageable chunks while preserving the semantic structure of crucial elements like tables, lists, and other HTML components.

Capabilities:
+ Preserves Important Elements: Maintains tables, lists, and other specified HTML elements.
+ Custom Handlers: Allows for custom handlers for specific HTML tags.
+ Semantic Integrity: Ensures the semantic meaning of the document is retained.
+ Normalization & Stopword Removal: Includes built-in normalization and stopword removal features.

Choosing the Right Splitter
+ Use HTMLHeaderTextSplitter When: You need to divide an HTML document based on its header hierarchy and retain metadata about the headers.
+ Use HTMLSectionSplitter When: You need to split the document into larger, more general sections, possibly based on custom tags or font sizes.
+ Use HTMLSemanticPreservingSplitter When: You need to break the document into chunks while preserving semantic elements like tables and lists, ensuring they remain intact and their context is maintained.

