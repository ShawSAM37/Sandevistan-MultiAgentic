AZURE_SEARCH_INDEX_NAME = "rotary-instruction-manuals"

APPROVED_INDEX_FIELDS = [
    "id",
    "content",
    "contentVector",
    "title",
    "titleVector",
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
    "citationPath",
]

RETRIEVABLE_FIELDS = [
    "id",
    "content",
    "title",
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
    "citationPath",
]

SEARCHABLE_TEXT_FIELDS = [
    "content",
    "title",
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
]

VECTOR_FIELDS = [
    "contentVector",
    "titleVector",
]

USER_FILTER_FIELDS = [
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
]

FACET_FIELDS = [
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
]

DEFAULT_VECTOR_DIMENSIONS = 1024

DEFAULT_TOP_K = 10
MAX_CONTEXT_CHARS = 50000
MAX_CHARS_PER_DOCUMENT = 10000
MAX_REVISION_COUNT = 1
MAX_SEARCH_PLANS = 3
MAX_LLM_CALLS_PER_REQUEST = 6
REQUEST_TIMEOUT_SECONDS = 60
