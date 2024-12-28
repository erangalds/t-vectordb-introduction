import pandas as pd
import psycopg
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = ollama.Client(host='http://host.docker.internal:11434')

def get_embedding(ollama_client, page_content):
    # Generate embeddings for the text
    response = ollama_client.embed(
        model='nomic-embed-text', 
        input=page_content  
    )
    return response['embeddings'][0]


loader = PyPDFLoader(
    "/sample-data/energy-policy.pdf",
)

documents = loader.load()

print(f'Number of Documents : {len(documents)}')

text_splitter_character_splitter = RecursiveCharacterTextSplitter( chunk_size=1000,chunk_overlap=200 )
docs = text_splitter_character_splitter.split_documents(documents) 

# Database connection parameters
dbname = 'pdf_rag'
user = 'postgres'
password = 'postgres'
host = 'vectordb-lab-postgres-db'
port = '5432'

# Connect to PostgreSQL database
conn_str = f'host={host} port={port} dbname={dbname} user={user} password={password}'
conn = psycopg.connect(conn_str)


# Insert data into PostgreSQL
with conn.cursor() as cur:
    for doc in docs:
        cur.execute("""
            INSERT INTO dev.pdf_rag_data (source, page_content, page_number, page_content_embeddings)
            VALUES (%s, %s, %s, %s)
            """,
            (doc.metadata['source'], doc.page_content, doc.metadata['page'], get_embedding(client,doc.page_content))
        )
    conn.commit()

conn.close()
