from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class DatasetSummary(BaseModel):
    """Summarizes a specific dataset and its most relevant evidence."""
    name: Optional[str] = Field(None, description="dataset_id from the retrieval")
    summary: Optional[str] = Field(
        None,
        description="A high-level synthesis and key takeaways of the dataset content relevant to the query. DO NOT restate or repeat the exact verbatim quote text."
    )
    quote: Optional[str] = Field(
        None,
        description="Exact verbatim excerpt copied directly from the top-ranked chunk of the dataset."
    )
    relevance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Retrieval-provided relevance score from 0.0 to 1.0. Never estimate this value."
    )

class Response(BaseModel):
    """The structured final response containing the answer and supporting evidence."""
    answer: Optional[str] = Field(
        None,
        description=(
            "A concise, factual answer strictly grounded in the retrieved context. "
            "Verify all facts and numerical claims directly against the retrieved chunks before stating them. "
            "If the context does not contain explicit data for a question (especially for missing or future temporal data, e.g. 2026 data), "
            "you MUST state clearly that the answer was not found in the documents. NEVER guess, extrapolate, or claim zero values."
        )
    )
    name_top: Optional[str] = Field(None, description="dataset_id from the top retrieval")
    evidence_status: Literal["supported", "insufficient"] = Field(
        "supported",
        description=(
            "Use 'supported' only when retrieved chunks directly answer the question. "
            "Use 'insufficient' when the requested fact or time period is absent."
        ),
    )
    supporting_datasets: List[DatasetSummary] = Field(
        default_factory=list,
        description=(
            "One item for every retrieved dataset, in retrieval rank order. Use only dataset_id values "
            "present in the context; each quote must be copied verbatim from that dataset's chunks."
        )
    )
