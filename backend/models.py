from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.constants import USER_FILTER_FIELDS


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    machine: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    manualType: str | None = None

    topK: int = Field(default=10, ge=1, le=50)
    searchMode: Literal["auto", "keyword", "vector", "hybrid"] = "auto"
    vectorTarget: Literal["auto", "contentVector", "titleVector", "both"] = "auto"
    useSemanticRanker: bool = True
    showDebug: bool = False


class Citation(BaseModel):
    citationId: int
    id: str
    title: str | None = None
    citationPath: str | None = None
    machine: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    manualType: str | None = None


class RetrievedDocument(BaseModel):
    id: str
    title: str | None = None
    content: str
    manualType: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    machine: str | None = None
    citationPath: str | None = None

    score: float | None = None
    rerankerScore: float | None = None
    searchMode: str | None = None
    vectorFieldsUsed: list[str] = []


class InputGuardrailResult(BaseModel):
    allowed: bool
    sanitizedQuestion: str
    riskLevel: Literal["low", "medium", "high"]
    reason: str


class DetectedEntities(BaseModel):
    machine: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    manualType: str | None = None
    errorCode: str | None = None


class QueryUnderstandingResult(BaseModel):
    intent: Literal[
        "maintenance_procedure",
        "operation_procedure",
        "safety",
        "troubleshooting",
        "error_code",
        "part_lookup",
        "specification",
        "document_lookup",
        "general_question",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    needsClarification: bool
    clarificationQuestion: str | None = None
    rewrittenQuery: str
    keywords: list[str] = []
    detectedEntities: DetectedEntities
    reason: str


class SearchPlan(BaseModel):
    query: str
    searchMode: Literal["keyword", "vector", "hybrid"]
    vectorFields: list[Literal["contentVector", "titleVector"]] = ["contentVector"]
    filters: dict[str, str] = {}
    top: int = Field(default=10, ge=1, le=50)
    useSemanticRanker: bool = True
    reason: str

    @model_validator(mode="after")
    def validate_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self


class RetrievalPlanningResult(BaseModel):
    searchPlans: list[SearchPlan]
    reason: str


class AnswerGenerationResult(BaseModel):
    answer: str
    usedCitationPaths: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    answerFound: bool


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str
    severity: Literal["low", "medium", "high"]


class GroundingCriticResult(BaseModel):
    grounded: bool
    groundingScore: float = Field(ge=0.0, le=1.0)
    requiresRevision: bool
    unsupportedClaims: list[UnsupportedClaim] = []
    missingCitations: bool = False
    recommendation: str


class SafetyCriticResult(BaseModel):
    safe: bool
    safetyScore: float = Field(ge=0.0, le=1.0)
    requiresRevision: bool
    safetyIssues: list[str] = []
    recommendation: str


class RevisionResult(BaseModel):
    revisedAnswer: str
    usedCitationPaths: list[str] = []
    answerFound: bool
    revisionReason: str


class AskResponse(BaseModel):
    requestId: str
    answer: str
    citations: list[Citation] = []
    retrievedDocuments: list[RetrievedDocument] = []

    intent: str | None = None
    rewrittenQuery: str | None = None

    grounded: bool | None = None
    safe: bool | None = None
    requiresRevision: bool = False
    responseType: str

    metadata: dict = {}

class DebugRetrievalRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=10, ge=1, le=50)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False
    showContent: bool = False

    @model_validator(mode="after")
    def validate_debug_retrieval_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class DebugContextRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=10, ge=1, le=50)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    @model_validator(mode="after")
    def validate_debug_context_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class DebugAnswerRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_answer_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class DebugGuardrailRequest(BaseModel):
    question: str = Field(min_length=1)


class GroundingCriticResult(BaseModel):
    grounded: bool
    requiresRevision: bool
    unsupportedClaims: list[str] = Field(default_factory=list)
    missingCitations: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DebugGroundingRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_grounding_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class RevisionResult(BaseModel):
    revisedAnswer: str
    usedCitationPaths: list[str] = Field(default_factory=list)
    revisionApplied: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

class DebugRevisionRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_revision_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class DebugRevisionRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_revision_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class SafetyCriticResult(BaseModel):
    safe: bool
    requiresRevision: bool
    safetyIssues: list[str] = Field(default_factory=list)
    missingWarnings: list[str] = Field(default_factory=list)
    unsafeOrUnsupportedInstructions: list[str] = Field(default_factory=list)
    inventedSafetyCriticalDetails: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

class DebugSafetyRequest(BaseModel):
    query: str = Field(min_length=1)

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_safety_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self

class DebugGraphAnswerRequest(BaseModel):
    query: str = Field(min_length=1)

    threadId: str | None = None
    userId: str | None = None

    searchMode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    vectorFields: list[Literal["contentVector", "titleVector"]] = Field(
        default_factory=lambda: ["contentVector"]
    )

    filters: dict[str, str] = Field(default_factory=dict)

    top: int = Field(default=3, ge=1, le=20)
    k: int = Field(default=50, ge=1, le=100)

    useSemanticRanker: bool = False

    maxContextChars: int | None = None
    maxCharsPerDocument: int | None = None

    includeDebugContext: bool = False

    @model_validator(mode="after")
    def validate_debug_graph_answer_filters(self):
        invalid_fields = set(self.filters.keys()) - set(USER_FILTER_FIELDS)
        if invalid_fields:
            raise ValueError(f"Unsupported filter fields for V1: {sorted(invalid_fields)}")
        return self
