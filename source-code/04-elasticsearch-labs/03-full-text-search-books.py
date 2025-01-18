from elasticsearch import Elasticsearch

# Elasticsearch credentials
username = 'your_username'
password = 'your_password'

# Initialize Elasticsearch client
es = Elasticsearch("http://vectordb-lab-elastic:9200")

def search_books_by_title(book_title):
    # Search query to find books by title
    query = {
        "query": {
            "match": {
                "title": book_title
            }
        }
    }

    response = es.search(index='mylibrary', body=query)
    hits = response['hits']['hits']

    print(f"\nSearching Title Query:\n{book_title}\n\nSearch Results:")
    for hit in hits:
        source = hit['_source']
        score = hit['_score']
        print(f"Title: {source['title']}\nAuthors: {', '.join(source['authors'])}\nRelevance Score: {score}\n")

if __name__ == "__main__":
    book_title = "cloud"  # Replace with the actual book title you want to search
    search_books_by_title(book_title)
