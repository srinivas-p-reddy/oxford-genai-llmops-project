import boto3
from botocore.exceptions import ClientError
import os
from typing import List, Dict, Union
from server.src.models.document import RetrievedDocument  # Import the Pydantic model
from server.src.config import Settings
from fastapi import Depends
import requests
import json
from server.src.config import settings
import opik
import openai
from openai import OpenAI

# client = OpenAI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@opik.track  # TODO: test if this works with async methods? I think it will.
def call_llm(prompt: str) -> Union[Dict, None]:
    """Call OpenAI's API to generate a response."""
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,  # Ensure this model is defined in settings
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_p=settings.top_p,
        )

        print("Successfully generated response")
        data = {"response": response.choices[0].message.content}
        data["response_tokens_per_second"] = (
            (response.usage.total_tokens / response.usage.completion_tokens)
            if hasattr(response, "usage")
            else None
        )
        print(f"call_llm returning {data}")
        print(f"data.response = {data['response']}")
        return data

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None  # TODO: error handling

@opik.track
async def generate_response(
    query: str,
    chunks: List[Dict],
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict:  # str:
    """
    Generate a response using an Ollama endpoint running locally, t
    his will be changed to allow for Bedrock later.

    Args:
        query (str): The user query.
        context (List[Dict]): The list of documents retrieved from the retrieval service.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Sampling temperature for the model.
    """
    QUERY_PROMPT = """
    You are a helpful AI language assistant, please use the following context to answer the query. Answer in English.
    Context: {context}
    Query: {query}
    Answer:
    """
    # Concatenate documents' summaries as the context for generation
    context = "\n".join([chunk["chunk"] for chunk in chunks])
    prompt = QUERY_PROMPT.format(context=context, query=query)
    print(f"calling call_llm ...")
    response = call_llm(prompt)
    print(f"generate_response returning {response}")
    return response  # now this is a dict.