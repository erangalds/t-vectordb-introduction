import psycopg
import ollama

client = ollama.Client(host='http://host.docker.internal:11434')

def get_embedding(ollama_client, title):
    # Generate embeddings for the text
    response = ollama_client.embed(
        model='nomic-embed-text', 
        input=title 
    )
    return response['embeddings'][0]
    
def fetch_book_titles():
    # Database connection parameters
    conn_params = {
        'dbname': 'mylibrary',           # Database name
        'user': 'postgres',         # PostgreSQL username
        'password': 'postgres',     # PostgreSQL password
        'host': 'vectordb-lab-postgres-db',  # Hostname (container name)
        'port': '5432'              # Port number
    }

    try:
        # Establishing the connection using psycopg3
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Executing SQL query to fetch all records from dev.books table
                cursor.execute("SELECT title FROM dev.books;")
                
                # Fetching all book titles
                book_titles = cursor.fetchall()
                
                # Printing the book titles
                print("Book Titles:")
                for title in book_titles:
                    print(f"- {title[0]}")

    except Exception as error:
        print(f"Error: {error}")
    # Returning the fetched books titles. 
    return book_titles
# Call the function to fetch and print book titles

def insert_vectorized_title(titles):
    # Database connection parameters
    conn_params = {
        'dbname': 'mylibrary',           # Database name
        'user': 'postgres',         # PostgreSQL username
        'password': 'postgres',     # PostgreSQL password
        'host': 'vectordb-lab-postgres-db',  # Hostname (container name)
        'port': '5432'              # Port number
    }
    try:
        # Establishing the connection using psycopg3
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                for title in titles:
                    # Update the dev.books table with the vectorized title 
                    print(f'\n\nVectorizing Title:\n{title[0]}\n')
                    vectorized_title = get_embedding(client, title[0])
                    print(f'Generated Embedding:\n{vectorized_title[0:5]}')
                    cursor.execute( "UPDATE dev.books SET vectorized_title = %s WHERE title = %s", (vectorized_title, title[0]) )
            conn.commit()
    except Exception as error:
        print(f'Error: {error}')

if __name__ == "__main__":
    retrieved_book_titles = fetch_book_titles()
    insert_vectorized_title(retrieved_book_titles)

