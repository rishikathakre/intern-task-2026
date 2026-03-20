"""Integration tests — require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def assert_valid_response(result):
    """Shared assertions for all integration tests."""
    assert result.corrected_sentence, "corrected_sentence must be non-empty"
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES, f"Invalid error_type: {error.error_type}"
        assert len(error.explanation) > 10, "Explanation should be substantive"
        assert len(error.original) > 0
        assert len(error.correction) > 0


# ---------------------------------------------------------------------------
# Language coverage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_correct_german_sentence():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_french_gender_agreement_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_japanese_particle_error_non_latin():
    """Non-Latin script: Japanese particle を → に."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_portuguese_spelling_and_grammar_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_correct_japanese_sentence():
    """Non-Latin correct sentence: should return is_correct=True."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は毎日学校に行きます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_arabic_sentence_non_latin():
    """Arabic (right-to-left non-Latin script) with an error."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="أنا ذهبت إلى المدرسة أمس.",
            target_language="Arabic",
            native_language="English",
        )
    )
    # Arabic sentence is correct (past tense with pronoun); just validate structure
    assert result.difficulty in VALID_DIFFICULTIES
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_italian_word_order_error():
    """Italian sentence with word order error — adjective before noun."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ho visto bella una ragazza.",
            target_language="Italian",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_response_time_under_30s():
    """Verify /feedback response is within the 30-second timeout gate."""
    import time
    start = time.time()
    result = await get_feedback(
        FeedbackRequest(
            sentence="The cat sat on the mat.",
            target_language="English",
            native_language="English",
        )
    )
    elapsed = time.time() - start
    assert elapsed < 30, f"Response took {elapsed:.1f}s — exceeds 30s limit"
    assert_valid_response(result)


@pytest.mark.asyncio
async def test_caching_returns_identical_result():
    """Same input twice must return the same response (from cache)."""
    from app.feedback import _cache
    _cache.clear()

    req = FeedbackRequest(
        sentence="Elle mange une pomme.",
        target_language="French",
        native_language="English",
    )
    result1 = await get_feedback(req)
    result2 = await get_feedback(req)

    assert result1.corrected_sentence == result2.corrected_sentence
    assert result1.is_correct == result2.is_correct
    assert result1.difficulty == result2.difficulty
