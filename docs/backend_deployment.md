# Backend Deployment

This document records the V1 backend deployment details for the Azure-native multi-agentic RAG system.

No secrets, keys, tokens, connection strings, tenant IDs, or subscription IDs should be stored in this file.

---

## Backend Runtime

The backend is deployed as an Azure Container App.

```text
Container App name:
ca-sandevistan-backend
Container Apps environment:
cae-sandevistan-rag
Resource group:
rg-sandvik-agentic-sandevistan-scRegion:
swedencentral
Backend Url:
https://ca-sandevistan-backend.graymushroom-28ea90b0.swedencentral.azurecontainerapps.io
Backend Image:
Registry:
acrsandevistan5606.azurecr.io
Image:
sandevistan-backend:phase1
Full image:
acrsandevistan5606.azurecr.io/sandevistan-backend:phase1
health endpoint:
/health

Managed identity:
id-sandevistan-aca
Next Backend Steps

Wire Azure AI Search configuration.
Wire Azure OpenAI configuration.
Add deep health checks.
Add Azure clients.
Add retrieval layer.
Add multi-agent graph.
