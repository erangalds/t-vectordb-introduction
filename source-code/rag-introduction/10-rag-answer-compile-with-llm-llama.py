import psycopg
from ollama import Client

def generate_embeddings(text):
    # Replace 'http://your-host-url:port' with the actual URL of your hosted service
    client = Client(host='http://host.docker.internal:11434')
    response = client.embed(model='nomic-embed-text', input=text)
    vetorized_search_query = response['embeddings'][0]
    print(f'Search Query:\n{text}')
    print(f'Vectorized Search Query:\n{vetorized_search_query[0:5]}')
    return vetorized_search_query

def similarity_search_euclidean_distance(query_text):
    # Database connection parameters
    conn_params = {
        'dbname': 'pdf_rag',           # Database name
        'user': 'postgres',         # PostgreSQL username
        'password': 'postgres',     # PostgreSQL password
        'host': 'vectordb-lab-postgres-db',  # Hostname (container name)
        'port': '5432'              # Port number
    }

    try:
        # Generate embedding for the query text
        query_embedding = generate_embeddings(query_text)

        # Establishing the connection using psycopg3
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Performing the similarity search
                cursor.execute(
                    """
                    SELECT source, page_content, page_number
                    FROM dev.pdf_rag_data
                    ORDER BY page_content_embeddings <-> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                
                # Printing the search results
                print("Similarity Search Results:")
                for result in results:
                    print(f"Source: {result[0]} Page Number: {result[2]}\nPage Content:\n{result[1]}")

    except Exception as error:
        print(f"Error: {error}")

def similarity_search_inner_product(query_text):
    # Database connection parameters
    conn_params = {
        'dbname': 'pdf_rag',           # Database name
        'user': 'postgres',         # PostgreSQL username
        'password': 'postgres',     # PostgreSQL password
        'host': 'vectordb-lab-postgres-db',  # Hostname (container name)
        'port': '5432'              # Port number
    }

    try:
        # Generate embedding for the query text
        query_embedding = generate_embeddings(query_text)

        # Establishing the connection using psycopg3
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Performing the similarity search
                cursor.execute(
                    """
                    SELECT source, page_content, page_number
                    FROM dev.pdf_rag_data
                    ORDER BY page_content_embeddings <#> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                
                # Printing the search results
                print("Similarity Search Results:")
                context_detail = ''
                for result in results:
                    #print(f"Source: {result[0]} Page Number: {result[2]}\nPage Content:\n{result[1]}")
                    context_detail += f"""
                    ####
                    Source 

                    {result[1]}

                    Source File= {result[0]}
                    Page Number= {result[2]}
                    
                    """
                
                #print(f'\n\n\n Source Details: \n{context_detail}')
                return context_detail
    except Exception as error:
        print(f"Error: {error}")

def generate_answer_from_llm(query_text,query_context):
    system_prompt = """
    You are a helpful assistant.
    Multiple Context information with the source details like the source file name and page number. 
    You need to carefully analyze each given context information and then answr the user query accordingly.
    When you compile the answer you have to refer the original source file name and page number which you used to generate the answer. 
    If you can't find the details to answer the querion, Please mention you don't know the answer.
    Final Answer should contain the Source File names and page numbers of the source details. 
    Final Answer should be formatted as below. 

    ##
    Answer: 

    ## References
    Source File Name: 
    Page Number:
    """
    
    user_prompt = f"""
    ###
    Context
    {query_context}

    ### User Query
    {query_text}
    """

    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ]

    client = Client(host='http://host.docker.internal:11434')

    response = client.chat(
        'llama3.2:1b',
        messages=messages
    )

    print(response.message.content)

# Call the function to perform similarity search
if __name__ == "__main__":
    query_text = "solar energy plan in the next few years"
    
    #print(f'\n\nUsing Euclidean Distance:\n\n')
    #similarity_search_euclidean_distance(query_text)
    #print(f'\n\nUsing Inner Product:\n\n')
    query_context = similarity_search_inner_product(query_text)

    print('\n\nFinal Answer: \n\n')
    generate_answer_from_llm(query_text=query_text,query_context=query_context)
