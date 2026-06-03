# V1 Requirements

V1 is an Azure-native multi-agentic RAG system.

## Azure-first rule

All major components must have an Azure execution path:

- FastAPI backend
- Streamlit UI
- Python scripts
- Evaluation jobs
- Logging
- Deployment workflow

## Approved index fields

V1 uses only:

- id
- content
- contentVector
- title
- titleVector
- manualType
- baseMachine
- serialNumber
- machine
- citationPath

The production index has 17 fields, but all other fields are unused for V1.
