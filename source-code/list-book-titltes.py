import psycopg

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

# Call the function to fetch and print book titles
if __name__ == "__main__":
    fetch_book_titles()
