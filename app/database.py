
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# Load environment variables from the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_connection():
    """
    Create and return a PostgreSQL database connection.
    """

    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")

    # Ensure database credentials are configured.
    if not all([db_name, db_user, db_password]):
        raise RuntimeError(
            "Database environment variables are missing. "
            "Please check your .env file."
        )

    connection = psycopg.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host="127.0.0.1",
        port=5432
    )

    return connection
