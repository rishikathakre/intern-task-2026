"""FastAPI application — language feedback endpoint."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.feedback import get_feedback
from app.models import FeedbackRequest, FeedbackResponse

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Language Feedback API",
    description="Analyzes learner-written sentences and provides structured language feedback.",
    version="1.1.0",
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


@app.get("/health")
async def health():
    """Health check endpoint — returns 200 when the service is running."""
    return {"status": "ok", "version": app.version}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Analyze a learner sentence and return structured language feedback."""
    return await get_feedback(request)
