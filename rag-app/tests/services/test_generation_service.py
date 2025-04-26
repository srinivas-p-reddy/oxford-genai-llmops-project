# example_prompts = ["tell me about quantum criticality for perovskites?",
#                    "what materials are often used along with perovskites?",
#                    "what electronic structure phenomena are important in recent perovskite papers?",
#                    "do any of the papers you know about mention band gaps of perovskites?"
#                    ]

import pytest
from typing import Dict, Union
from server.src.services.generation_service import generate_response

# Leverages the mock's from conftest.py
@pytest.mark.asyncio
async def test_generate_response_basic(
    mock_query, mock_chunks, mock_config, mock_generate_response
):
    # Mock the response from the LLM
    """Test the basic functionality of the generate_response function.

    This test verifies that the function correctly processes a basic query
    with provided mock document chunks and configuration. It ensures that
    the response includes pertinent information from both the query and context.

    Mocks:
        mock_generate_response: Simulates the LLM's response to return a predefined
            response containing 'perovskites' and 'solar cells'.

    Assertions:
        - The response is either a dictionary or None.
        - The response includes the keyword 'perovskites'.
        - The response references 'solar cells', indicating it utilizes context
          from the retrieved chunks.
    """
    mock_generate_response.return_value = {
        "response": "Here is information about perovskites: They are used in solar cells.",
        "eval_count": 100,
        "eval_duration": 0.1,
    }

    # Call the function under test
    response = generate_response(mock_query, mock_chunks, **mock_config)
    print(response)
    # Assertions
    assert isinstance(response, Union[Dict, None]), "Response should be a Dict or None."
    assert (
        "perovskites" in response["response"]
    ), "Response should contain relevant query content."
    assert (
        "solar cells" in response["response"]
    ), "Response should refer to context from retrieved chunks."


@pytest.mark.asyncio
async def test_generate_response_empty_chunks(
    mock_query, mock_config, mock_generate_response
):
    # Mock response for empty chunks
    """
    Test the generate_response function with an empty list of chunks.

    This test verifies that the function returns a specific message when provided with
    an empty list of document chunks, ensuring that it handles cases where no context
    is available for response generation.

    Mocks:
        mock_generate_response: Mocks the LLM's response to return a predefined
        message indicating no relevant information was found.

    Assertions:
        - The response matches the expected message for empty chunks.
    """
    mock_generate_response.return_value = (
        "No relevant information found for perovskites in solar cells."
    )

    # Call the function with an empty list of chunks
    response = generate_response(mock_query, [], **mock_config)

    # Assertions
    assert (
        response == "No relevant information found for perovskites in solar cells."
    ), "Should return a specific message for empty chunks."


@pytest.mark.asyncio
async def test_generate_response_high_temperature(
    mock_query, mock_chunks, mock_generate_response
):
    # Mock response for high temperature setting
    """
    Args:
        mock_query: The mock query input for testing.
        mock_chunks: The mock document chunks for testing.
        mock_generate_response: A mocked version of the generate_response function.

    Test the generate_response function with a high temperature setting.

    This test checks whether the function behaves as expected when given a high
    temperature setting, ensuring it returns a string response and respects
    the max_tokens limit.
    """
    mock_generate_response.return_value = (
        "Perovskites might revolutionize solar cells with surprising applications."
    )

    # Call the function with a high temperature setting
    response = asyncio.run(generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=1.5
    ))

    # Assertions
    assert isinstance(response, str), "Response should still be a string."
    assert len(response.split()) <= 150, "Response should respect the max_tokens limit."


@pytest.mark.asyncio
async def test_generate_response_long_query(mock_chunks, mock_generate_response):
    """Test generate_response with a long query string

    This test checks if the generate_response function can handle a long query
    without errors and produce a valid response.

    Mocks:
        mock_generate_response: Mocks the response from the LLM to return a
            predefined string containing 'Perovskites' and 'solar cells'.

    Assertions:
        - The response contains the keyword 'Perovskites'.
        - The response does not exceed the max_tokens limit.
    """
    # Simulate a long query by repeating the word 'Perovskites'
    long_query = "Perovskites " * 100

    # Mock the response expected from the LLM for the long query
    mock_generate_response.return_value = (
        "Perovskites are materials used in solar cells."
    )

    # Call the generate_response function with the long query
    response = asyncio.run(generate_response(
        long_query, mock_chunks, max_tokens=150, temperature=0.7
    ))

    # Assertions
    assert "Perovskites" in response, "Response should handle long query without error."
    assert len(response.split()) <= 150, "Response should not exceed max_tokens."


@pytest.mark.asyncio
async def test_generate_response_with_multiple_chunks(
    mock_query, mock_chunks, mock_generate_response
):
    """
    Test generate_response with multiple chunks.

    This test verifies that the generate_response function can handle
    multiple chunks and produce a valid response.

    Mocks:
        mock_generate_response: Mocks the response from the LLM to return a
            predefined string containing 'Perovskites', 'solar cells',
            'unique properties', and 'efficiency has improved'.

    Assertions:
        - The response contains the keywords 'Perovskites', 'solar cells',
          'unique properties', and 'efficiency has improved'.
    """
    mock_generate_response.return_value = (
        "Perovskites are used in solar cells and have unique properties. "
        "Their efficiency has recently improved."
    )

    # Call the function with multiple chunks
    response = asyncio.run(generate_response(
        mock_query, mock_chunks, max_tokens=150, temperature=0.7
    ))

    # Assertions
    assert "used in solar cells" in response
    assert "unique properties" in response
    assert "efficiency has improved" in response
