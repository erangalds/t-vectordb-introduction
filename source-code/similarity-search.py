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
        'dbname': 'mylibrary',           # Database name
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
                    SELECT title, vectorized_title
                    FROM dev.books
                    ORDER BY vectorized_title <-> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                
                # Printing the search results
                print("Similarity Search Results:")
                for result in results:
                    print(f"Title: {result[0]}")

    except Exception as error:
        print(f"Error: {error}")

def similarity_search_inner_product(query_text):
    # Database connection parameters
    conn_params = {
        'dbname': 'mylibrary',           # Database name
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
                    SELECT title, vectorized_title
                    FROM dev.books
                    ORDER BY vectorized_title <#> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                
                # Printing the search results
                print("Similarity Search Results:")
                for result in results:
                    print(f"Title: {result[0]}")

    except Exception as error:
        print(f"Error: {error}")


# Call the function to perform similarity search
if __name__ == "__main__":
    query_text = "cloud solutions"
    
    print(f'\n\nUsing Euclidean Distance:\n\n')
    similarity_search_euclidean_distance(query_text)
    print(f'\n\nUsing Inner Product:\n\n')
    similarity_search_inner_product(query_text)
