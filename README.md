# Language Feedback API

An LLM-powered language feedback API that analyzes learner-written sentences and returns structured correction feedback. Built for [Pangea Chat](https://pangea.chat) — an NSF-funded language learning platform.

---

## Quick Start

### Local (without Docker)

```bash
# 1. Clone and enter the repo
git clone <your-fork-url>
cd intern-task-2026

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Start the server
uvicorn app.main:app --reload

# 6. Test it
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

### With Docker

```bash
cp .env.example .env
# Edit .env with your API key
docker compose up --build
```

The service starts on port **8000**. Health check: `GET /health`.

---

## Running Tests

```bash
# Unit tests (no API key needed — uses mocked LLM responses)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires OPENAI_API_KEY in .env)
pytest tests/test_feedback_integration.py -v

# All tests
pytest -v
```

> Tests run inside Docker too (`docker compose exec feedback-api pytest -v`), because all test dependencies are installed in the image.

---

## API Reference

### `POST /feedback`

**Request:**

```json
{
  "sentence": "Yo soy fue al mercado ayer.",
  "target_language": "Spanish",
  "native_language": "English"
}
```

**Response:**

```json
{
  "corrected_sentence": "Yo fui al mercado ayer.",
  "is_correct": false,
  "errors": [
    {
      "original": "soy fue",
      "correction": "fui",
      "error_type": "conjugation",
      "explanation": "You mixed two verb forms. 'Soy' is present tense of 'ser' (to be), and 'fue' is past tense of 'ir' (to go). You only need 'fui' (I went)."
    }
  ],
  "difficulty": "A2"
}
```

Allowed `error_type` values: `grammar`, `spelling`, `word_choice`, `punctuation`, `word_order`, `missing_word`, `extra_word`, `conjugation`, `gender_agreement`, `number_agreement`, `tone_register`, `other`

Allowed `difficulty` values: `A1`, `A2`, `B1`, `B2`, `C1`, `C2` (CEFR scale)

### `GET /health`

Returns `{"status": "ok", "version": "1.1.0"}` with HTTP 200.

---

## Design Decisions

### Model: `gpt-4o-mini` at `temperature=0`

- **Why gpt-4o-mini?** It achieves near-GPT-4-level linguistic accuracy at ~10× lower cost and latency, well within the 30-second response limit. For a production language learning app handling thousands of students, cost per call matters enormously.
- **Why temperature=0?** We want deterministic, accurate grammar analysis — not creative variation. Temperature 0 minimizes hallucinations and gives consistent results across repeated identical inputs.

### Prompt Engineering

The system prompt is designed around six explicit rules that the model must follow:

1. **Correct sentence handling** — If there are no errors, `is_correct` must be `true`, `errors` must be `[]`, and `corrected_sentence` must be character-for-character identical to the input.
2. **Error taxonomy** — The prompt lists all 12 allowed `error_type` values explicitly, with examples of what each covers.
3. **Minimal correction** — The model is instructed to apply the fewest edits possible, preserving the learner's voice. This is educationally important: learners should see their own style reflected back with targeted fixes, not a complete rewrite.
4. **Native-language explanations** — Explanations are written in the learner's native language so beginners can actually understand the feedback. For non-Latin scripts (Japanese, Arabic, Chinese), the prompt instructs the model to preserve target-language characters in explanations for clarity.
5. **CEFR difficulty grounded in benchmarks** — The prompt provides concrete A1–C2 sentence examples so the model assigns difficulty based on the sentence's structural complexity, *not* whether it has errors.
6. **JSON-only output** — The prompt explicitly says "no prose before or after, no markdown fences." Combined with `response_format={"type": "json_object"}`, this virtually eliminates parse failures.

### Response Caching

Identical `(sentence, target_language, native_language)` triples return the same response from an in-process SHA-256-keyed TTL cache (1-hour expiry). This matters for production:

- Language learners often make the same mistakes repeatedly
- Conversation activities in Pangea Chat may send the same sentence for group members
- Avoids redundant API calls and latency entirely for repeated inputs

The cache is in-process (dict), so it resets on restart. A Redis-backed cache would be the next step for multi-instance deployment, but adds operational complexity not warranted for an intern task.

### Retry Logic

The API call retries up to 3 times with exponential backoff (1s → 2s → 4s) on:
- `429 RateLimitError` — OpenAI throttling under load
- `5xx APIError` — transient server errors

Client errors (4xx except 429) are not retried — they indicate a bad request that won't succeed on retry.

### Defensive Output Validation

Even with `response_format=json_object`, models occasionally return out-of-spec enum values (e.g., `"Conjugation"` instead of `"conjugation"`). Two layers of defense:
1. `feedback.py` normalizes `difficulty` (e.g., `"b1"` → `"B1"`) and coerces invalid `error_type` to `"other"` before constructing the Pydantic model
2. `models.py` Pydantic `@field_validator`s provide a second layer, coercing any invalid values that slip through

This ensures near-100% schema compliance even if the model occasionally deviates.

### Error Handling

A FastAPI global exception handler catches any unhandled exception and returns a structured `{"detail": "..."}` JSON 500 response, so the client always gets JSON (not an HTML error page).

---

## Test Strategy

| File | Type | Coverage |
|------|------|----------|
| `test_feedback_unit.py` | Unit (no API key) | 10 tests: conjugation, correct sentence, multiple errors, Japanese (non-Latin), missing_word, number_agreement, spelling, non-English native language, cache hit behavior, validator coercion |
| `test_feedback_integration.py` | Integration (real API) | 10 tests: Spanish, German, French, Japanese (error + correct), Portuguese, Arabic, Italian, 30s response time gate, caching idempotency |
| `test_schema.py` | Schema validation | 9 tests: valid/invalid request and response, all example I/O pairs |

Unit tests use `unittest.mock` to replace the OpenAI client — they run instantly without any API key and validate that the application layer correctly parses and returns structured responses.

Integration tests hit the real API and validate linguistic correctness across multiple languages and script systems. The `assert_valid_response()` helper is shared across all integration tests to ensure consistent structural checks.

---

## Production Considerations

If this were shipping at Pangea Chat's scale, the next iterations would be:

- **Redis cache** — Share the cache across workers/instances, persist across restarts
- **Async batching** — Group concurrent requests hitting the same sentence and resolve them all with a single API call
- **Streaming** — For the UI, stream the response back so the learner sees feedback appear progressively
- **Rate limiting** — Per-user request quotas to prevent abuse
- **Observability** — Structured logging with request IDs, latency histograms, cache hit rate metrics
- **Fallback model** — If GPT-4o-mini is down, fall back to GPT-3.5-turbo for continuity
