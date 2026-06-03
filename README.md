
# Sandevistan Multi-Agentic RAG

Azure-native multi-agentic RAG system for rotary instruction manuals.

## V1 Scope

The system is designed to run fully in Azure, including:

- FastAPI backend
- Streamlit frontend
- Azure AI Search retrieval
- Azure OpenAI chat and embedding calls
- Python utility scripts
- Evaluation jobs
- Logs and observability
- CI/CD deployment workflow

## Azure Resource Group

\`\`\`text
rg-sandvik-agentic-sandevistan-sc
\`\`\`

## Azure Region

\`\`\`text
swedencentral
\`\`\`

## GitHub Repository

\`\`\`text
https://github.com/ShawSAM37/Sandevistan-MultiAgentic
\`\`\`

## Approved V1 Azure AI Search Fields

The production index has 17 fields, but V1 uses only:

\`\`\`text
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
\`\`\`

All other index fields are considered unused for V1.

## V1 Architecture Summary

\`\`\`text
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
\`\`\`

## Current Phase

\`\`\`text
Phase 1 — Skeleton and contracts
\`\`\`
