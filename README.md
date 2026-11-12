# Building Semantic Memory with Microsoft Foundry and Azure Cosmos DB for NoSQL

A runnable Python lab by Victor Vicente Pereira Torrealba. It extracts a memory from a synthetic patient conversation, stores it with a vector embedding, and retrieves it from a differently worded question.

This project adapts Microsoft Learn’s [Implement semantic memory with vector storage](https://learn.microsoft.com/en-us/training/modules/aaai-design-multi-agent-memory-azure-cosmos-db/3-implement-semantic-memory-vector-storage?pivots=text) training unit into an end-to-end implementation verified on Azure. It accompanies my [Microsoft Certified: Multi-Agent AI Solutions Expert credential](https://learn.microsoft.com/en-us/users/victorvicentepereiratorrealba-6127/credentials/e93b105f1f8311bf). The implementation demonstrates service-principal authentication, vector storage, and semantic retrieval.

## Azure setup

1. In the same Microsoft Foundry resource, deploy `gpt-5-mini` for extraction and `text-embedding-3-small` for embeddings. Use the actual deployment names in your configuration. The sample uses the Foundry Model Inference `/models` endpoint.
2. Create an Azure Cosmos DB for NoSQL account and enable **Vector Search for NoSQL API**. Then create the database and container with the following settings.
3. Configure a Microsoft Entra service principal with permission to invoke the Foundry deployments. Grant that principal the Cosmos DB NoSQL **Cosmos DB Built-in Data Contributor** data-plane role scoped to the lab container.

| Cosmos DB setting | Value |
|---|---|
| Database ID | `clinical-memory-db` |
| Container ID | `patient-memories` |
| Partition key | `/patient_id` |
| Vector path and index path | `/embedding` |
| Vector data type | `float32` |
| Dimensions | `1536` |
| Distance function | `cosine` |
| Vector index | `quantizedFlat` |
| Quantizer type | Product |
| Provisioned throughput used in the lab | Manual, 400 RU/s |

Use the **Application (client) ID** for `AZURE_CLIENT_ID`. For the Cosmos DB role assignment, use the **Object ID** of the service principal under **Microsoft Entra ID → Enterprise applications**. You can resolve it from the client ID with `az ad sp show --id "<application-client-id>" --query id -o tsv`.

From Azure Cloud Shell PowerShell, set the active subscription to the one that owns the Cosmos DB account and run the following with your values:

```powershell
$resourceGroup = "<cosmos-resource-group>"
$accountName = "<cosmos-account-name>"
$principalId = "<service-principal-object-id>"
$accountId = az cosmosdb show --resource-group $resourceGroup --name $accountName --query id -o tsv

az cosmosdb sql role assignment create `
  --resource-group $resourceGroup `
  --account-name $accountName `
  --principal-id $principalId `
  --role-definition-id "$accountId/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002" `
  --scope "$accountId/dbs/clinical-memory-db/colls/patient-memories"
```

[Microsoft Learn: Cosmos DB data-plane role assignments](https://learn.microsoft.com/en-us/azure/cosmos-db/how-to-connect-role-based-access-control)

## Run in VS Code PowerShell

Extract the ZIP and open the folder containing all three Python files. In that folder:

```powershell
python -m venv .venv
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fill in the placeholders in `.env`. Use the Cosmos DB **account URI** and the Foundry inference URL ending in `/models`. Keep the filled `.env` local and commit only `.env.example`.

```powershell
.\.venv\Scripts\python.exe store.py
.\.venv\Scripts\python.exe retrieve.py
```

- **`extract.py`** extracts structured observations. Run it by itself if you want to check extraction separately.
- **`store.py`** imports and calls `extract_memories` from `extract.py`. Run `store.py` directly for the combined extraction and storage step. It embeds each observation and writes it to the `patient-12345` partition. Repeating this step can add more demo items.
- **`retrieve.py`** imports the embedding client from `extract.py` and the container from `store.py`. It embeds the scheduling question, searches the same patient partition, and patches the matching items’ access count and timestamp. Importing either module leaves its standalone demo block idle.

## Verified result

In the live run, Azure Data Explorer returned the stored preference **“Patient prefers morning appointments.”** and reported **1,536 embedding dimensions**. The retrieval script found that memory using the question **“What time of day should we schedule the patient's next visit?”**. The terminal showed one retrieved memory, a score of **0.580**, and an access count of **1**. Scores and generated text can vary by run.

The query returns the closest matches using `ORDER BY VectorDistance(...)` and a top-K limit. A fixed `> 0.7` filter would have excluded the verified match. Retrieval updates access metadata with `patch_item()` so the stored embedding and partition key are preserved.

[Microsoft Learn: Vector search in Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-search)
