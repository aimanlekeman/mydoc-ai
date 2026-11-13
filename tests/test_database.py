
from app.database import get_connection


def test_database_connection():
    print("Connecting to MyDoc AI database...")

    try:
        with get_connection() as connection:
            result = connection.execute(
                "SELECT version();"
            ).fetchone()

            print("\nDatabase connection successful!")
            print(result[0])

    except Exception as error:
        print("\nDatabase connection failed!")
        print(error)


if __name__ == "__main__":
    test_database_connection()
