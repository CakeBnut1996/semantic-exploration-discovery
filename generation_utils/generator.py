from generation_utils.llm_client import LLMClient
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence, Type, Union
from pydantic import BaseModel

from generation_utils.schema import DatasetSummary, Response


# Include this version in cache keys so changing generation guardrails invalidates old responses.
GUARDRAIL_VERSION = "2026-08-31-v2"

# Reuse a cached response for seven days before deleting and regenerating it.
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Match four-digit years from 1900 through 2199 for temporal evidence checks.
YEAR_PATTERN = re.compile(r"\b(?:19|20|21)\d{2}\b")

# Convert written counts in requests such as "top three" into numeric result limits.
COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

class StudentGenerator:
    def __init__(self, provider: str, model_name: str):
        """
        The Student uses the LLMClient for text generation.
        """
        self.llm = LLMClient(provider, model_name)

    @staticmethod
    def _cache_path(provider: str, model_name: str, query: str, context: str, schema: Type[BaseModel]) -> Path:
        payload = json.dumps(
            {
                "guardrail_version": GUARDRAIL_VERSION,
                "provider": provider.lower(),
                "model": model_name,
                "query": query,
                "context": context,
                "schema": schema.model_json_schema(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = sha256(payload.encode("utf-8")).hexdigest()
        return Path(__file__).resolve().parent.parent / ".generation_cache" / f"{key}.json"

    @staticmethod
    def _missing_query_years(query: str, context: str) -> list[str]:
        requested = set(YEAR_PATTERN.findall(query))
        evidence_text = context
        try:
            decoded = json.loads(context)
            if isinstance(decoded, list):
                evidence_text = " ".join(
                    chunk
                    for dataset in decoded
                    if isinstance(dataset, dict)
                    for chunk in dataset.get("chunks", [])
                    if isinstance(chunk, str)
                )
        except (TypeError, ValueError):
            pass
        available = set(YEAR_PATTERN.findall(evidence_text))
        return sorted(requested - available)

    def generate(self, query: str, context: str, schema: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel]:
        system_instr = (
            "You are a helpful assistant. Answer strictly based on the context provided.\n"
            "CRITICAL RULES:\n"
            "1. Fact Verification & Groundedness: Verify all facts and numerical claims directly against the retrieved context before answering. "
            "If the retrieved context does not contain explicit data for a question (especially out-of-scope or future temporal queries like 2026 data), "
            "you MUST state clearly that the information was not found in the documents. NEVER guess, extrapolate, or claim zero values.\n"
            "2. Temporal Absence: A year mentioned in the question is not evidence. The same year must occur in a retrieved chunk before you may make a claim about it. "
            "Treat absent years as unavailable data, even if the user repeats or insists on the question.\n"
            "3. Non-Redundancy: In dataset summaries, provide high-level context and takeaways; do NOT quote, paraphrase sentence-by-sentence, or restate the quote field.\n"
            "4. Evidence Integrity: Use every retrieved dataset exactly once and in the supplied rank order. Copy quotes character-for-character from that dataset's chunks. "
            "Do not invent dataset IDs or relevance scores."
        )
        prompt = f"Context:\n{context}\n\nUser question: {query}\n"

        if schema:
            missing_years = self._missing_query_years(query, context)
            if missing_years and issubclass(schema, Response):
                years = ", ".join(missing_years)
                return schema(
                    answer=f"The retrieved documents do not contain data for {years}, so this question cannot be answered from the available evidence.",
                    name_top=None,
                    evidence_status="insufficient",
                    supporting_datasets=[],
                )

            cache_path = self._cache_path(self.llm.provider, self.llm.model_name, query, context, schema)
            if cache_path.exists():
                if time.time() - cache_path.stat().st_mtime < CACHE_TTL_SECONDS:
                    return schema.model_validate_json(cache_path.read_text(encoding="utf-8"))
                cache_path.unlink()

            result = self.llm.generate_structured(prompt, schema, system_instruction=system_instr)
            if isinstance(result, BaseModel):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(result.model_dump_json(), encoding="utf-8")
            return result

        response = self.llm.generate_text(prompt, system_instruction=system_instr)
        if response is None:
            return "Error: The LLM returned no response."
        return response


def serialize_ranked_context(ranked_data: Sequence[Any]) -> str:
    """Create stable, explicit JSON context instead of Python object reprs."""
    payload = []
    for dataset in ranked_data:
        payload.append({
            "dataset_id": dataset.dataset_id,
            "rank": len(payload) + 1,
            "retrieval_relevance": round(float(dataset.top_score), 8),
            "source_title": dataset.source_title,
            "chunks": [chunk["text"] for chunk in dataset.top_chunks],
        })
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def select_ranked_context(query: str, ranked_data: Sequence[Any]) -> list[Any]:
    """Honor an explicit ``top N`` request without letting the model vary N."""
    count_pattern = "|".join(COUNT_WORDS)
    match = re.search(rf"\btop\s+(\d+|{count_pattern})\b", query, re.IGNORECASE)
    if not match:
        return list(ranked_data)
    raw_count = match.group(1).lower()
    requested_count = int(raw_count) if raw_count.isdigit() else COUNT_WORDS[raw_count]
    return list(ranked_data[:requested_count])


def _summary_repeats_quote(summary: str, quote: str) -> bool:
    if summary.casefold() == quote.casefold():
        return True
    summary_tokens = re.findall(r"\w+", summary.casefold())
    quote_tokens = set(re.findall(r"\w+", quote.casefold()))
    if len(summary_tokens) < 5:
        return False
    overlap = sum(token in quote_tokens for token in summary_tokens) / len(summary_tokens)
    return overlap >= 0.8


def ground_response(response: Response, ranked_data: Sequence[Any]) -> Response:
    """Force citations, scores, and excerpts to match deterministic retrieval."""
    if not ranked_data or response.evidence_status == "insufficient":
        response.name_top = None
        response.supporting_datasets = []
        return response

    response.name_top = ranked_data[0].dataset_id
    generated = {item.name: item for item in response.supporting_datasets if item.name}
    grounded: list[DatasetSummary] = []

    for dataset in ranked_data:
        candidate = generated.get(dataset.dataset_id)
        chunks = [chunk["text"] for chunk in dataset.top_chunks]
        # The displayed excerpt is the complete top-ranked chunk. Allowing the
        # model to select a substring could reintroduce mid-sentence excerpts.
        quote = chunks[0] if chunks else None

        summary = candidate.summary.strip() if candidate and candidate.summary else None
        if summary and quote and _summary_repeats_quote(summary, quote):
            summary = "This source contains evidence relevant to the question; see the verbatim excerpt below."

        grounded.append(DatasetSummary(
            name=dataset.dataset_id,
            summary=summary or "This source contains evidence relevant to the question; see the verbatim excerpt below.",
            quote=quote,
            relevance_score=max(0.0, min(1.0, float(dataset.top_score))),
        ))

    response.supporting_datasets = grounded
    return response
