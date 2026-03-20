"""LLM interaction for language feedback — prompt engineering & resilience."""

import asyncio
import hashlib
import json
import logging
import time

from openai import AsyncOpenAI, APIError, RateLimitError

from app.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process response cache (sentence + target + native → response)
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[FeedbackResponse, float]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_key(request: FeedbackRequest) -> str:
    raw = f"{request.sentence}||{request.target_language}||{request.native_language}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(key: str) -> FeedbackResponse | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    response, ts = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return response


def _set_cached(key: str, response: FeedbackResponse) -> None:
    _cache[key] = (response, time.time())


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert language teacher and linguist. A language learner has written a sentence in their target language.
Your job is to analyze the sentence, identify ALL errors (no matter how minor), and return precise, structured feedback.

CRITICAL RULES — follow every one of these exactly:

1. CORRECT SENTENCES: If the sentence has no errors at all, return:
   - is_correct: true
   - errors: [] (empty array)
   - corrected_sentence: the EXACT original sentence, character-for-character

2. ERROR IDENTIFICATION: Find EVERY error, including:
   - Verb conjugation mistakes (wrong tense, person, number)
   - Gender and number agreement (articles, adjectives, pronouns)
   - Word order violations specific to the target language
   - Missing required words (articles, prepositions, particles)
   - Extra/redundant words
   - Spelling errors
   - Punctuation errors
   - Wrong word choice or register

3. CORRECTED SENTENCE: Apply the MINIMUM edits needed — preserve the learner's intended meaning and voice. Do not rephrase or rewrite unless absolutely necessary.

4. ERRORS ARRAY: For each error:
   - "original": the EXACT substring from the input sentence that is wrong
   - "correction": the correct replacement for that substring
   - "error_type": MUST be one of these exact strings only:
     grammar, spelling, word_choice, punctuation, word_order, missing_word,
     extra_word, conjugation, gender_agreement, number_agreement, tone_register, other
   - "explanation": 1–2 sentences, friendly and educational, written in the learner's NATIVE language.
     For non-Latin scripts (Japanese, Arabic, Chinese, Korean, etc.) keep the target-language characters
     in the explanation for clarity, but write the explanation prose in the native language.

5. DIFFICULTY (CEFR): Rate sentence complexity, NOT correctness. Use these benchmarks:
   - A1: Isolated words, very basic phrases ("Hola", "Mi nombre es Juan")
   - A2: Simple present/past, common vocabulary, short sentences
   - B1: Compound sentences, multiple tenses, subordinate clauses
   - B2: Complex grammar, nuanced vocabulary, passive voice, conditionals
   - C1: Advanced structures, idiomatic expressions, subtle register choices
   - C2: Native-level complexity, literary/academic language

6. JSON OUTPUT: You MUST respond with ONLY a valid JSON object — no prose before or after, no markdown fences.
   The JSON must exactly match this schema:
   {
     "corrected_sentence": "<string>",
     "is_correct": <boolean>,
     "errors": [
       {
         "original": "<string>",
         "correction": "<string>",
         "error_type": "<one of the allowed types>",
         "explanation": "<string in native language>"
       }
     ],
     "difficulty": "<A1|A2|B1|B2|C1|C2>"
   }
"""

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds (doubles on each retry)


async def _call_with_retry(
    client: AsyncOpenAI,
    user_message: str,
    timeout: float = 25.0,
) -> dict:
    """Call the OpenAI API with exponential-backoff retry on transient errors."""
    delay = _RETRY_DELAY
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,          # deterministic: we want accuracy, not creativity
                    max_tokens=1024,        # cap cost; responses are typically ~300 tokens
                ),
                timeout=timeout,
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except RateLimitError as exc:
            last_error = exc
            logger.warning("Rate limit hit (attempt %d/%d), retrying in %.1fs", attempt + 1, _MAX_RETRIES, delay)
            await asyncio.sleep(delay)
            delay *= 2

        except APIError as exc:
            last_error = exc
            if exc.status_code and exc.status_code >= 500:
                logger.warning("OpenAI server error %s (attempt %d/%d)", exc.status_code, attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise  # client errors (4xx except 429) are not retryable

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("JSON decode failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
            # No sleep — try immediately with same prompt

    raise RuntimeError(f"All {_MAX_RETRIES} attempts failed: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Return structured language feedback for the given learner sentence."""
    cache_key = _cache_key(request)
    cached = _get_cached(cache_key)
    if cached is not None:
        logger.debug("Cache hit for key %s", cache_key[:12])
        return cached

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from env automatically

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence to analyze: {request.sentence}"
    )

    data = await _call_with_retry(client, user_message)

    # Normalize: ensure difficulty and error_type values are uppercase/valid
    # (defensive coercion in case the model slightly deviates)
    difficulty_map = {"a1": "A1", "a2": "A2", "b1": "B1", "b2": "B2", "c1": "C1", "c2": "C2"}
    raw_difficulty = str(data.get("difficulty", "")).strip()
    data["difficulty"] = difficulty_map.get(raw_difficulty.lower(), raw_difficulty)

    valid_error_types = {
        "grammar", "spelling", "word_choice", "punctuation", "word_order",
        "missing_word", "extra_word", "conjugation", "gender_agreement",
        "number_agreement", "tone_register", "other",
    }
    for error in data.get("errors", []):
        if error.get("error_type", "").lower() not in valid_error_types:
            error["error_type"] = "other"

    result = FeedbackResponse(**data)
    _set_cached(cache_key, result)
    return result
