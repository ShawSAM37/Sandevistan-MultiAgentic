from pydantic import BaseModel


class MultiAgentRagState(BaseModel):
    request_id: str
    runtime: str = "azure"

    question: str
    machine: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    manualType: str | None = None

    topK: int = 10
    searchMode: str = "auto"
    vectorTarget: str = "auto"
    useSemanticRanker: bool = True
    showDebug: bool = False

    allowed: bool | None = None
    sanitizedQuestion: str | None = None
    riskLevel: str | None = None
    guardrailReason: str | None = None

    intent: str | None = None
    intentConfidence: float | None = None
    needsClarification: bool = False
    clarificationQuestion: str | None = None
    rewrittenQuery: str | None = None
    keywords: list[str] = []
    detectedEntities: dict = {}

    searchPlans: list[dict] = []

    retrievedDocuments: list[dict] = []
    searchModesUsed: list[str] = []
    vectorFieldsUsed: list[str] = []
    filtersUsed: dict = {}

    context: str | None = None
    citations: list[dict] = []
    usedDocuments: list[dict] = []
    skippedDocuments: list[dict] = []
    contextCharCount: int = 0

    draftAnswer: str | None = None
    revisedAnswer: str | None = None
    finalAnswer: str | None = None
    answerFound: bool = False
    answerConfidence: float | None = None

    grounded: bool | None = None
    safe: bool | None = None
    requiresRevision: bool = False

    groundingCriticResult: dict | None = None
    safetyCriticResult: dict | None = None
    revisionGroundingCriticResult: dict | None = None
    revisionSafetyCriticResult: dict | None = None

    revisionCount: int = 0
    responseType: str | None = None

    agentsExecuted: list[str] = []
    steps: list[dict] = []
    latencyMs: int | None = None
    tokenUsage: dict = {}
    errors: list[dict] = []
    metadata: dict = {}
