"""Unit tests — run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.feedback import get_feedback, _cache, _cache_key
from app.models import FeedbackRequest, FeedbackResponse


def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_request(**kwargs) -> FeedbackRequest:
    defaults = dict(sentence="Hola mundo.", target_language="Spanish", native_language="English")
    defaults.update(kwargs)
    return FeedbackRequest(**defaults)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_with_conjugation_error():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(sentence="Yo soy fue al mercado ayer.")
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    mock_response = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
        )
        result = await get_feedback(request)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == request.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_gender_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(
            sentence="La chat noir est sur le table.",
            target_language="French",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


# ---------------------------------------------------------------------------
# Non-Latin script (Japanese)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_non_latin_japanese():
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む takes particle に for location.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
        )
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "grammar"
    assert "に" in result.errors[0].correction


# ---------------------------------------------------------------------------
# Missing word error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_missing_word():
    mock_response = {
        "corrected_sentence": "Je veux aller à l'école.",
        "is_correct": False,
        "errors": [
            {
                "original": "aller école",
                "correction": "aller à l'école",
                "error_type": "missing_word",
                "explanation": "You need the preposition 'à' before 'école'.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(sentence="Je veux aller école.", target_language="French")
        result = await get_feedback(request)

    assert result.is_correct is False
    assert result.errors[0].error_type == "missing_word"


# ---------------------------------------------------------------------------
# Number agreement error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_number_agreement():
    mock_response = {
        "corrected_sentence": "Los niños son inteligentes.",
        "is_correct": False,
        "errors": [
            {
                "original": "niños son inteligente",
                "correction": "niños son inteligentes",
                "error_type": "number_agreement",
                "explanation": "Adjectives in Spanish agree in number. 'inteligentes' (plural) matches 'niños' (plural).",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(sentence="Los niños son inteligente.", target_language="Spanish")
        result = await get_feedback(request)

    assert result.errors[0].error_type == "number_agreement"


# ---------------------------------------------------------------------------
# Spelling error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_spelling_error():
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Present/gift' in Portuguese is spelled 'presente' with an 's'.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = _make_request(sentence="Eu quero comprar um prezente.", target_language="Portuguese")
        result = await get_feedback(request)

    assert result.errors[0].error_type == "spelling"


# ---------------------------------------------------------------------------
# Non-English native language (explanation in native language)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_native_language_explanation():
    mock_response = {
        "corrected_sentence": "Ich gehe morgen ins Kino.",
        "is_correct": False,
        "errors": [
            {
                "original": "gehe morgen im Kino",
                "correction": "gehe morgen ins Kino",
                "error_type": "grammar",
                "explanation": "Bei 'ins Kino' wird 'in' mit 'das' zu 'ins' verschmolzen (Kontraktion).",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=_mock_completion(mock_response))

        request = FeedbackRequest(
            sentence="Ich gehe morgen im Kino.",
            target_language="German",
            native_language="German",  # German learner studying German
        )
        result = await get_feedback(request)

    # Explanation should be in German (native language)
    assert len(result.errors[0].explanation) > 0
    assert result.is_correct is False


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_returns_same_response():
    _cache.clear()  # start fresh

    mock_response = {
        "corrected_sentence": "Hola, ¿cómo estás?",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        create_mock = AsyncMock(return_value=_mock_completion(mock_response))
        instance.chat.completions.create = create_mock

        request = _make_request(sentence="Hola, ¿cómo estás?")
        result1 = await get_feedback(request)
        result2 = await get_feedback(request)  # should hit cache

    # API should only be called once even though we called get_feedback twice
    assert create_mock.call_count == 1
    assert result1 == result2


# ---------------------------------------------------------------------------
# Schema compliance — difficulty validator coerces invalid values
# ---------------------------------------------------------------------------

def test_difficulty_validator_coerces_invalid():
    resp = FeedbackResponse(
        corrected_sentence="test",
        is_correct=True,
        errors=[],
        difficulty="Z9",  # invalid — should be coerced to A1
    )
    assert resp.difficulty == "A1"


# ---------------------------------------------------------------------------
# Schema compliance — error_type validator coerces invalid values
# ---------------------------------------------------------------------------

def test_error_type_validator_coerces_invalid():
    from app.models import ErrorDetail
    err = ErrorDetail(
        original="foo",
        correction="bar",
        error_type="not_a_real_type",
        explanation="some explanation",
    )
    assert err.error_type == "other"
