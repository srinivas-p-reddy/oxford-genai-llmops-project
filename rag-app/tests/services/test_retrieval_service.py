import pytest
from server.src.services.retrieval_service import retrieve_top_k_chunks
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

# Database connection configuration
db_config = {
    "dbname": os.environ.get("POSTGRES_DB"),
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "host": os.environ.get("POSTGRES_HOST"),
    "port": os.environ.get("POSTGRES_PORT"),
}

# Test function for the retrieval service
@pytest.mark.skip(reason="Skipping DB connection test in CI environment")
def test_retrieve_top_k_chunks():
    query = "perovskite"
    top_k = 5

    try:
        documents = retrieve_top_k_chunks(query, top_k, db_config)

        # Assertions
        assert isinstance(documents, list)
        assert len(documents) <= top_k

        for doc in documents:
            assert "id" in doc
            assert "title" in doc
            assert "summary" in doc
            assert "similarity_score" in doc
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
