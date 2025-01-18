from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loaders
# Document Loaders load data into the standard Langchain Document format
# Each Document Loader has its own specific parameters, but they can all be invoked in the same way. 
# Below is an example of using the PyPDFLoader from langchain which uses pypdf package.

loader = PyPDFLoader(
    "/sample-data/energy-policy.pdf",
)

documents = loader.load()

print(f'Number of Documents : {len(documents)}')

text_splitter_character_splitter = RecursiveCharacterTextSplitter( chunk_size=1000,chunk_overlap=200 )
docs = text_splitter_character_splitter.split_documents(documents) 

print(f'Number of Document Chunks: {len(docs)}')

print(f'\n\nContent of the first document chunk:\n{docs[0]}')

print(f'\n\nContent of the Second document chunk:\n{docs[1]}')