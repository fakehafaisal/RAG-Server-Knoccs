import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
HOST = os.getenv("PGVECTOR_HOST")
PORT = os.getenv("PGVECTOR_PORT")
DBNAME = os.getenv("PGVECTOR_DATABASE")
USER = os.getenv("PGVECTOR_USER")
PASSWORD = os.getenv("PGVECTOR_PASSWORD")

# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        sslmode="require",         
        connect_timeout=10,
        dbname=DBNAME
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    print(f"Failed to connect: {e}")

