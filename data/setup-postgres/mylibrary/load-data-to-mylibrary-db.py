import pandas as pd
import psycopg

# Read Excel file
excel_file = '/sample-data/mylibrary/mylibrary.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Convert Authors column to array type
df['Authors'] = df['Authors'].apply(lambda x: x.split(','))

# Convert Tags column to array type
df['Tags'] = df['Tags'].apply(lambda x: x.split(','))

# Database connection parameters
dbname = 'mylibrary'
user = 'postgres'
password = 'postgres'
host = 'vectordb-lab-postgres-db'
port = '5432'

# Connect to PostgreSQL database
conn_str = f'host={host} port={port} dbname={dbname} user={user} password={password}'
conn = psycopg.connect(conn_str)

# Insert data into PostgreSQL
with conn.cursor() as cur:
    for index, row in df.iterrows():
        cur.execute("""
            INSERT INTO dev.books (ISBN, Title, Edition, PublishedYear, Publisher, Authors, Tags)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (row['ISBN'], row['Title'], row['Edition'], row['Published Year'], row['Publisher'], row['Authors'], row['Tags'])
        )
    conn.commit()

conn.close()
