# Architecture

## V1 Runtime

Streamlit UI on Azure calls FastAPI backend on Azure.

The backend owns:

- multi-agent orchestration
- Azure OpenAI calls
- Azure AI Search retrieval
- context building
- citations
- critics
- revision
- logging
- evaluation endpoints

## Multi-agent flow

User question
→ input guardrail
→ query understanding
→ retrieval planning
→ Azure AI Search retrieval
→ context builder
→ answer generation
→ grounding critic
→ safety critic
→ optional one-pass revision
→ final answer
