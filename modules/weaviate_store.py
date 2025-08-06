import os
import uuid
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from typing import List
from dotenv import load_dotenv

load_dotenv()  # Load .env file with OPENAI_API_KEY


class WeaviateStore:
    def __init__(self, collection_name: str = "ChunkEmbedding"):
        self.collection_name = collection_name
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        )
        self.collection = self._ensure_collection()

    def _ensure_collection(self):
        if self.collection_name not in self.client.collections.list_all():
            print(f"🛠️ Creating collection `{self.collection_name}`")
            return self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT)
                ],
                vectorizer_config=None  # External embeddings (OpenAI, etc.)
            )
        else:
            return self.client.collections.get(self.collection_name)

    # Add to your `WeaviateStore` class
    def clear_collection(self):
        self.collection.data.delete_many(where={})

    
    def store_embeddings(self, texts: List[str], embeddings: List[List[float]], source: str = "pdf"):
        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings must be the same length.")

        with self.collection.batch.fixed_size(batch_size=100) as batch:
            for text, vector in zip(texts, embeddings):
                batch.add_object(
                    properties={"text": text, "source": source},
                    vector=vector,
                    uuid=uuid.uuid4()
                )
                if batch.number_errors > 10:
                    print("⛔ Batch import stopped due to too many errors.")
                    break

        failed = self.collection.batch.failed_objects
        if failed:
            print(f"⚠️ Failed objects: {len(failed)}")
            print(f"First error: {failed[0]}")
        else:
            print(f"✅ Successfully stored {len(texts)} chunks in Weaviate.")

    def close(self):
        self.client.close()
