import psycopg

def list_databases():
    # Database connection parameters
    conn_params = {
        'dbname': 'postgres',       # Default database to connect to
        'user': 'postgres',         # PostgreSQL username
        'password': 'postgres',     # PostgreSQL password
        'host': 'vectordb-lab-postgres-db',  # Hostname or IP address (container name)
        'port': '5432'              # Port number
    }

    try:
        # Establishing the connection using psycopg3
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Query to list databases
                cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
                
                # Fetching all databases
                databases = cursor.fetchall()
                
                # Printing the database names
                print("Available databases:")
                for db in databases:
                    print(f"- {db[0]}")

    except Exception as error:
        print(f"Error: {error}")

# Call the function to list databases
if __name__ == "__main__":
    list_databases()
