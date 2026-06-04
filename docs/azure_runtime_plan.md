# Azure Runtime Plan

This document lists the Azure resources used by the V1 Azure-native multi-agentic RAG system.

No secrets, keys, connection strings, tenant IDs, or subscription IDs should be stored in this file.

---

## Resource Group

```text
rg-sandvik-agentic-sandevistan-sc
````

## Region

```text
swedencentral
```

***

## Azure Resources

| Purpose                  | Resource Name        | Azure Type               |
| ------------------------ | -------------------- | ------------------------ |
| Container image registry | acrsandevistan5606   | Azure Container Registry |
| Logs workspace           | log-sandevistan-rag  | Log Analytics Workspace  |
| App monitoring           | appi-sandevistan-rag | Application Insights     |
| Eval/result storage      | stsandevistan36945   | Azure Storage Account    |
| Secrets/configuration    | kv-sandevistan-4921  | Azure Key Vault          |

***

## Runtime Model

V1 is Azure-native. The target runtime is:

```text
Streamlit UI on Azure
    ↓
FastAPI backend on Azure
    ↓
Controlled multi-agent graph
    ↓
Azure OpenAI + Azure AI Search
    ↓
Grounded answer with citations
    ↓
Grounding and safety critics
    ↓
One controlled revision loop
```

***

## Approved Azure AI Search Index Contract

The production Azure AI Search index is:

```text
rotary-instruction-manuals
```

The production index contains 17 fields, but V1 uses only these 10 fields:

```text
id
content
contentVector
title
titleVector
manualType
baseMachine
serialNumber
machine
citationPath
```

All other index fields are unused for V1.

***

## Embedding Contract

Document vectors in the production index were generated using:

```text
text-embedding-3-large
dimensions = 1024
```

Therefore, runtime query embeddings must also use:

```text
text-embedding-3-large
dimensions = 1024
```

The backend must explicitly request 1024 dimensions for query embeddings.

***

## Retrieval Contract

V1 retrieval must only return these fields:

```text
id
content
title
manualType
baseMachine
serialNumber
machine
citationPath
```

Vector fields may be queried but must not be returned:

```text
contentVector
titleVector
```

User-facing filters are limited to:

```text
manualType
baseMachine
serialNumber
machine
```

***

## Storage Usage

The storage account is used for:

```text
evaluation outputs
eval reports
failure case reports
future run artifacts
```

The V1 eval output container is:

```text
eval-results
```

***

## Key Vault Usage

Key Vault is used for production secrets and configuration such as:

```text
Azure OpenAI endpoint/key or managed identity references
Azure AI Search endpoint/key or managed identity references
storage configuration
application secrets
```

The Key Vault is configured with:

```text
soft delete enabled
purge protection enabled
90-day retention
Azure RBAC authorization
```

***

## Logging and Observability

Application logs and traces should flow to:

```text
Application Insights
Log Analytics Workspace
```

The backend should emit structured logs for:

```text
request_id
agent step
retrieval plan
search mode
vector fields used
filters used
latency
token usage
grounding result
safety result
revision result
fallback reason
```

***

## Security Notes

Do not commit:

```text
.env
.azure-local-notes.txt
API keys
connection strings
storage keys
tenant IDs
subscription IDs
Key Vault secret values
Azure OpenAI keys
Azure AI Search keys
```

Use Azure RBAC and managed identity where possible.

For V1, secrets may be stored in Azure Key Vault or Azure Container App/App Service secret configuration.

***

## Current Project Phase

```text
Phase 1 — Skeleton and contracts
```

Next planned infrastructure step:

```text
Create Azure Container Apps environment for backend and frontend hosting.
```


---

## Backend Container App

The FastAPI backend skeleton is deployed to Azure Container Apps.

```text
Container App:
ca-sandevistan-backend
````

```text
Backend URL:
https://ca-sandevistan-backend.graymushroom-28ea90b0.swedencentral.azurecontainerapps.io
```

```text
Container image:
acrsandevistan5606.azurecr.io/sandevistan-backend:phase1
```

Current health endpoint:

```text
/health
```

Current deployment state:

```text
Azure-hosted backend skeleton is live.
Azure AI Search and Azure OpenAI runtime configuration are not yet wired.
```

