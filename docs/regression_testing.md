# Sandevistan Regression Testing
This document explains how to run production regression tests for the Sandevistan `/ask` endpoint.

## Purpose
The regression script verifies that the production LangGraph-backed `/ask` endpoint keeps the expected V1 behavior after changes to:
- prompts
- graph nodes
- model routing
- Azure OpenAI deployments
- Azure AI Search retrieval
- safety and grounding critics
- deployment configuration

The script is located at:
```text
scripts/test_ask_endpoint.py
```
