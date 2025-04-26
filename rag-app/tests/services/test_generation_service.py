import pytest
import asyncio
from typing import Dict, Union
from server.src.services.generation_service import generate_response

@pytest.mark.asyncio
async def test_generate_response_basic(
    mock_query, mock_chunks, mock_config, mock_generate_response
):
    mock_generate_response.return_value = {
        "response": "Here is information about perovskites: They are used in solar cells.",
        "eval_count": 100,
        "eval_duration": 0.1,
    }

    # Call the function under test
    response = await generate_response(mock_query, mock_chunks, **mock_config)

    # Assertions
    assert isinstance(response, (Dict, type(None))), "Response should be a Dict or None."
    assert "perovskites" in response["response"], "Response should contain relevant query content."
    assert "solar cells" in response["response"], "Response should refer to context from retrieved chunks."

@pytest.mark.asyncio
async def test_generate_response_empty_chunks(
    mock_query, mock_config, mock_generate_response
):
    mock_generate_response.return_value = (
        "No relevant information found for perovskites in solar cells."
    )

    response = await generate_response(mock_query, [], **mock_config)

    assert response == "No relevant information found for perovskites in solar cells.", \
        "Should return a specific message for empty chunks."

@pytest.mark.asyncio
async def test_generate_response_high_temperature(
    mock_query, mock_chunks, mock_generate_response
):
    mock_generate_response.return_value = (
        "Perovskites might revolutionize solar cells with surprising applications."
    )

    response = await generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=1.5
    )

    assert isinstance(response, str), "Response should still be a string."
    assert len(response.split()) <= 150, "Response should respect the max_tokens limit."

@pytest.mark.asyncio
async def test_generate_response_long_query(mock_chunks, mock_generate_response):
    long_query = "Perovskites " * 100

    mock_generate_response.return_value = (
        "Perovskites are materials used in solar cells."
    )

    response = await generate_response(
        long_query, mock_chunks, max_tokens=150, temperature=0.7
    )

    assert "Perovskites" in response, "Response should handle long query without error."
    assert len(response.split()) <= 150, "Response should not exceed max_tokens."

@pytest.mark.asyncio
async def test_generate_response_with_multiple_chunks(
    mock_query, mock_chunks, mock_generate_response
):
    mock_generate_response.return_value = (
        "Perovskites are used in solar cells and have unique properties. "
        "Their efficiency has recently improved."
    )

    response = await generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=0.7
    )

    assert "used in solar cells" in response
    assert "unique properties" in response
    assert "efficiency has improved" in response
