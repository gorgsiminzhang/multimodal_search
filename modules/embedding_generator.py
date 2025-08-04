import os
from typing import List
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env file with OPENAI_API_KEY

class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        self.client = OpenAI(api_key=self.api_key)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using the OpenAI API.
        Each chunk must be a string under 8192 tokens.
        """
        try:
            # OpenAI recommends batching for performance
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return []

