"""Retrieve semantic memories for one patient from Azure Cosmos DB."""

import os
from datetime import datetime, timezone

from extract import embed_client
from store import container


def retrieve_relevant_memories(
    patient_id: str, session_context: str, top_k: int = 10
) -> list[dict]:
    """Return the closest stored memories and update their access metadata."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    response = embed_client.embed(
        input=[session_context],
        model=os.environ["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    )
    embedding = response.data[0].embedding
    if len(embedding) != 1536:
        raise ValueError(f"Expected 1536 embedding dimensions, got {len(embedding)}.")

    # Use the vector ordering documented for Cosmos DB.
    query = """
        SELECT TOP @top_k c.id, c.content, c.importance, c.memory_type,
            c.timestamp,
            VectorDistance(c.embedding, @query_embedding) AS similarity_score
        FROM c
        WHERE c.patient_id = @patient_id
        ORDER BY VectorDistance(c.embedding, @query_embedding)
    """
    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@patient_id", "value": patient_id},
        {"name": "@query_embedding", "value": embedding},
    ]
    items = list(
        container.query_items(
            query=query,
            parameters=parameters,
            partition_key=patient_id,
        )
    )

    accessed_at = datetime.now(timezone.utc).isoformat()
    for item in items:
        # Patch the stored document to preserve its embedding and partition key.
        updated = container.patch_item(
            item=item["id"],
            partition_key=patient_id,
            patch_operations=[
                {"op": "incr", "path": "/access_count", "value": 1},
                {"op": "set", "path": "/last_accessed", "value": accessed_at},
            ],
        )
        item["access_count"] = updated["access_count"]
        item["last_accessed"] = updated["last_accessed"]

    return items


if __name__ == "__main__":
    memories = retrieve_relevant_memories(
        patient_id="patient-12345",
        session_context="What time of day should we schedule the patient's next visit?",
        top_k=5,
    )
    print(f"Retrieved {len(memories)} memory item(s).")
    for memory in memories:
        print(
            f"[{memory['memory_type']}] {memory['content']} "
            f"(importance: {memory['importance']}, "
            f"score: {memory['similarity_score']:.3f}, "
            f"access count: {memory['access_count']})"
        )
