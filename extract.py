import json
import os

from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
model_name = os.environ["AZURE_MODEL_DEPLOYMENT_NAME"]

credential = ClientSecretCredential(
    tenant_id=os.environ["AZURE_TENANT_ID"],
    client_id=os.environ["AZURE_CLIENT_ID"],
    client_secret=os.environ["AZURE_CLIENT_SECRET"],
)
credential_scopes = ["https://cognitiveservices.azure.com/.default"]

chat_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=credential_scopes,
)

embed_client = EmbeddingsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=credential_scopes,
)

memory_extraction_prompt = """Analyze this conversation interaction and extract observations worth remembering for future sessions with this patient.

Conversation Context:
{conversation_history}

Recent Agent Response:
{agent_response}

Extract 0-3 memorable observations that meet these criteria:
- Reveals patient preferences, concerns, or care needs
- Documents clinical patterns or responses to treatment
- Records decisions made or plans established
- Contains information relevant for future interactions

For each observation, provide:
1. A concise statement (1-2 sentences)
2. An importance score (1-10, where 10 is critical clinical information)
3. The observation type (preference, clinical_pattern, concern, decision, or context)

Output as JSON:
{{
  "observations": [
    {{"statement": "...", "importance": 8, "type": "clinical_pattern"}},
    {{"statement": "...", "importance": 6, "type": "preference"}}
  ]
}}

If nothing memorable was discussed, return: {{"observations": []}}"""


def extract_memories(
    conversation_history: str, agent_response: str
) -> list[dict]:
    """Extract memorable observations from an agent interaction."""
    prompt = memory_extraction_prompt.format(
        conversation_history=conversation_history,
        agent_response=agent_response,
    )
    response = chat_client.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format="json_object",
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The chat model returned an empty response.")

    result = json.loads(content)
    return result["observations"]


if __name__ == "__main__":
    observations = extract_memories(
        conversation_history="The patient prefers morning appointments.",
        agent_response="I will remember that preference.",
    )
    print(json.dumps(observations, indent=2, ensure_ascii=False))