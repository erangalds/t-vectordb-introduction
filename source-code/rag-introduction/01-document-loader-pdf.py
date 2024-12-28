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

