"""Store extracted observations as vector memories in Azure Cosmos DB."""

import os
from datetime import datetime, timezone
from uuid import uuid4

from azure.cosmos import CosmosClient

from extract import credential, embed_client, extract_memories

cosmos_client = CosmosClient(
    url=os.environ["COSMOS_DB_ENDPOINT"],
    credential=credential,
)
database = cosmos_client.get_database_client("clinical-memory-db")
container = database.get_container_client("patient-memories")


def store_memory(patient_id: str, observation: dict) -> None:
    """Embed one observation and store it in the patient's partition."""
    statement = observation["statement"]
    response = embed_client.embed(
        input=[statement],
        model=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    )
    embedding = response.data[0].embedding
    if len(embedding) != 1536:
        raise ValueError(
            f"Expected 1536 embedding dimensions for patient-memories, "
            f"but the deployment returned {len(embedding)}."
        )

    memory_doc = {
        "id": str(uuid4()),
        "patient_id": patient_id,
        "content": statement,
        "importance": observation["importance"],
        "memory_type": observation["type"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding": embedding,
        "access_count": 0,
        "last_accessed": None,
    }
    container.upsert_item(memory_doc)
    print(f"Stored memory {memory_doc['id']} ({len(embedding)} dimensions).")


if __name__ == "__main__":
    observations = extract_memories(
        conversation_history="The patient prefers morning appointments.",
        agent_response="I will remember that preference.",
    )
    for observation in observations:
        store_memory("patient-12345", observation)
    print(f"Stored {len(observations)} observation(s).")
