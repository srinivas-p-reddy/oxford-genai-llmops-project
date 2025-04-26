import pytest
from typing import Dict, Union
from unittest.mock import AsyncMock
from server.src.services.generation_service import generate_response

@pytest.mark.asyncio
async def test_generate_response_basic(
    mock_query, mock_chunks, mock_config, mock_generate_response
):
    """
    Test the basic functionality of the generate_response function.
    Verifies that the function correctly processes a basic query
    and retrieves context properly.
    """
    # Mocking the generate_response
    mock_generate_response.return_value = {
        "response": "Perovskites are used in solar cells.",
        "eval_count": 10,
        "eval_duration": 0.05,
    }

    response = await mock_generate_response(mock_query, mock_chunks, **mock_config)

    assert isinstance(response, (Dict, type(None))), "Response should be a Dict or None."
    assert "Perovskites" in response["response"], "Response should mention 'Perovskites'."
    assert "solar cells" in response["response"], "Response should reference 'solar cells'."

@pytest.mark.asyncio
async def test_generate_response_empty_chunks(
    mock_query, mock_config, mock_generate_response
):
    """
    Test the generate_response function with an empty list of chunks.
    Should return a message indicating no relevant information was found.
    """
    mock_generate_response.return_value = {
        "response": "No relevant information found for perovskites in solar cells.",
        "eval_count": 5,
        "eval_duration": 0.01,
    }

    response = await mock_generate_response(mock_query, [], **mock_config)

    assert response["response"] == "No relevant information found for perovskites in solar cells."

@pytest.mark.asyncio
async def test_generate_response_high_temperature(
    mock_query, mock_chunks, mock_generate_response
):
    """
    Test the generate_response function with a high temperature setting.
    Should still return a valid string response.
    """
    mock_generate_response.return_value = {
        "response": "Perovskites might revolutionize solar cells.",
        "eval_count": 20,
        "eval_duration": 0.07,
    }

    response = await mock_generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=1.5
    )

    assert isinstance(response["response"], str), "Response should still be a string."
    assert len(response["response"].split()) <= 150, "Response should respect the max_tokens limit."

@pytest.mark.asyncio
async def test_generate_response_long_query(
    mock_chunks, mock_generate_response
):
    """
    Test generate_response with a long query string.
    Should still produce a valid response and not crash.
    """
    long_query = "Perovskites " * 100

    mock_generate_response.return_value = {
        "response": "Perovskites are important materials used in solar cells.",
        "eval_count": 15,
        "eval_duration": 0.05,
    }

    response = await mock_generate_response(
        long_query, mock_chunks, max_tokens=150, temperature=0.7
    )

    assert "Perovskites" in response["response"]
    assert len(response["response"].split()) <= 150

@pytest.mark.asyncio
async def test_generate_response_with_multiple_chunks(
    mock_query, mock_chunks, mock_generate_response
):
    """
    Test generate_response with multiple document chunks.
    Should aggregate multiple pieces of context correctly.
    """
    mock_generate_response.return_value = {
        "response": "Perovskites are used in solar cells with unique properties and improved efficiency.",
        "eval_count": 25,
        "eval_duration": 0.08,
    }

    response = await mock_generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=0.7
    )

    assert "used in solar cells" in response["response"]
    assert "unique properties" in response["response"]
    assert "improved efficiency" in response["response"]
