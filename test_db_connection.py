"""
Simple database connection test script.
Use this to verify your PostgreSQL/pgvector connection is working.
"""

import psycopg2
from src.config import DatabaseConfig


def test_connection():
    """Test database connection using centralized config"""
    print("Testing database connection...")
    print(f"Host: {DatabaseConfig.HOST}")
    print(f"Port: {DatabaseConfig.PORT}")
    print(f"Database: {DatabaseConfig.DATABASE}")
    print(f"User: {DatabaseConfig.USER}")
    print()

    try:
        connection = psycopg2.connect(**DatabaseConfig.get_connection_params())
        print("✓ Connection successful!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Test query
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        print(f"✓ Current database time: {result[0]}")

        # Check pgvector extension
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            print(f"✓ pgvector extension version: {result[0]}")
        else:
            print("⚠ pgvector extension not installed")

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("✓ Connection closed.")
        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    return True


if __name__ == "__main__":
    test_connection()
