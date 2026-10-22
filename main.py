"""FastAPI backend for Azure AI Generative Studio.

Implements chat, image analysis, and prompt-to-SVG-to-PNG generation against
Azure OpenAI's Responses API. Azure credentials are read from environment vars.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("azure-generative-studio")

app = FastAPI(
    title="Azure AI Generative Studio API",
    description="Chat, image analysis, and vector-style PNG creation using Azure OpenAI.",
    version="1.0.0",
)

_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_CHAT_MESSAGES = 20
MAX_MESSAGE_CHARS = 12_000
REQUEST_TIMEOUT_SECONDS = 90.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


def get_responses_url() -> str:
    """Normalize supported Azure OpenAI endpoint forms to Responses API URL."""
    endpoint = (os.getenv("AZURE_ENDPOINT") or "").strip().rstrip("/")
    if not endpoint:
        raise HTTPException(status_code=500, detail="AZURE_ENDPOINT is missing in .env.")

    if endpoint.endswith("/openai/v1/responses"):
        return endpoint
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/responses"
    return f"{endpoint}/openai/v1/responses"


def extract_text(data: dict[str, Any]) -> str:
    """Extract plain text from an OpenAI Responses API payload."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                value = part.get("text", "")
                if isinstance(value, str) and value:
                    chunks.append(value)
    return "\n".join(chunks).strip()


async def call_azure(input_payload: list[dict[str, Any]]) -> dict[str, Any]:
    """Call Azure OpenAI Responses API without exposing the key to clients."""
    api_key = (os.getenv("AZURE_API_KEY") or "").strip()
    model = (os.getenv("AZURE_MODEL") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="AZURE_API_KEY is missing in .env.")
    if not model:
        raise HTTPException(status_code=500, detail="AZURE_MODEL is missing in .env.")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                get_responses_url(),
                headers={"Content-Type": "application/json", "api-key": api_key},
                json={"model": model, "input": input_payload},
            )
    except httpx.TimeoutException as exc:
        logger.warning("Azure request timed out")
        raise HTTPException(status_code=504, detail="Azure request timed out. Please try again.") from exc
    except httpx.RequestError as exc:
        logger.exception("Could not reach Azure OpenAI")
        raise HTTPException(status_code=502, detail="Could not connect to the Azure OpenAI endpoint.") from exc

    try:
        data = response.json()
    except ValueError:
        data = {}

    if not response.is_success:
        # Do not log API keys or request headers.
        message = (
            data.get("error", {}).get("message")
            if isinstance(data, dict) and isinstance(data.get("error"), dict)
            else None
        )
        message = message or f"Azure API error {response.status_code}"
        logger.warning("Azure API returned status %s", response.status_code)
        status = response.status_code if 400 <= response.status_code < 500 else 502
        raise HTTPException(status_code=status, detail=message)

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Azure returned an unexpected response.")
    return data


def ensure_supported_image(upload: UploadFile, raw: bytes) -> None:
    """Validate MIME type and real file format to avoid trusting file extensions."""
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a PNG, JPG, or WebP image.")
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MiB or smaller.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            actual_format = image.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    mime_to_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    if actual_format != mime_to_format[upload.content_type]:
        raise HTTPException(status_code=400, detail="Image contents do not match the uploaded file type.")


def sanitize_svg(svg_text: str) -> str:
    """Extract and reject active/external SVG content before rasterization."""
    svg_text = re.sub(r"^\s*```(?:svg|xml)?\s*", "", svg_text, flags=re.IGNORECASE)
    svg_text = re.sub(r"\s*```\s*$", "", svg_text).strip()
    start = svg_text.find("<svg")
    end = svg_text.rfind("</svg>")
    if start < 0 or end < 0 or end < start:
        raise HTTPException(
            status_code=502,
            detail="The model did not return SVG markup. Try a more specific prompt.",
        )
    svg = svg_text[start : end + len("</svg>")]

    unsafe_svg = re.compile(
        r"<\s*(?:script|foreignObject|iframe|object|embed|image)\b"
        r"|<!DOCTYPE|<!ENTITY|\bon\w+\s*="
        r"|\b(?:href|xlink:href)\s*="
        r"|url\s*\(\s*['\"]?\s*(?:https?:|file:|data:)",
        re.IGNORECASE,
    )
    if unsafe_svg.search(svg):
        raise HTTPException(
            status_code=502,
            detail="The returned SVG contained unsupported content. Please try again.",
        )

    root_end = svg.find(">")
    if root_end < 0:
        raise HTTPException(status_code=502, detail="The model returned malformed SVG markup.")
    if not re.search(r"\bxmlns\s*=", svg[:root_end]):
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "Azure AI Generative Studio API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": os.getenv("AZURE_MODEL", "not configured")}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    history: list[dict[str, str]] = []
    for message in request.messages:
        if message.role not in {"user", "assistant"}:
            continue
        content = message.content.strip()
        if not content:
            continue
        history.append({"role": message.role, "content": content[:MAX_MESSAGE_CHARS]})
    history = history[-MAX_CHAT_MESSAGES:]

    if not history:
        raise HTTPException(status_code=400, detail="Please provide a message.")

    data = await call_azure(
        [{"role": "system", "content": "You are a helpful AI assistant."}, *history]
    )
    reply = extract_text(data)
    if not reply:
        raise HTTPException(status_code=502, detail="Azure returned no text response.")
    return {"reply": reply}


@app.post("/api/vision")
async def vision(image: UploadFile = File(...)) -> dict[str, str]:
    raw = await image.read(MAX_IMAGE_BYTES + 1)
    try:
        ensure_supported_image(image, raw)
        encoded = base64.b64encode(raw).decode("ascii")
        data_url = f"data:{image.content_type};base64,{encoded}"
        data = await call_azure(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe this image and explain its main visual details.",
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ]
        )
        reply = extract_text(data)
        return {"reply": reply or "The model did not return a description."}
    finally:
        await image.close()


@app.post("/api/generate-image")
async def generate_image(request: GenerateImageRequest) -> Response:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Please describe the image you want.")

    data = await call_azure(
        [
            {
                "role": "system",
                "content": (
                    "You create attractive vector illustrations from user prompts. "
                    "Return ONLY valid SVG markup, with no explanation or Markdown. "
                    "The SVG must have width='1024', height='1024', and a viewBox. "
                    "Include xmlns='http://www.w3.org/2000/svg'. "
                    "Use inline vector shapes, paths, and text only. "
                    "Do not use scripts, foreignObject, embedded images, external files, "
                    "links, external URLs, or animations."
                ),
            },
            {"role": "user", "content": f"Create an SVG illustration of: {prompt}"},
        ]
    )
    svg = sanitize_svg(extract_text(data))

    try:
        # Pillow's SVG support depends on optional backends; CairoSVG gives consistent
        # vector rasterization while preserving the SVG safety checks above.
        import cairosvg
        png = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=1024,
            output_height=1024,
        )
    except ImportError as exc:
        logger.error("CairoSVG is not installed")
        raise HTTPException(
            status_code=500,
            detail="PNG conversion dependency is missing. Install requirements.txt and restart.",
        ) from exc
    except Exception as exc:
        logger.exception("SVG-to-PNG conversion failed")
        raise HTTPException(
            status_code=502,
            detail="Could not convert the generated SVG to PNG. Please try another prompt.",
        ) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="generated.png"',
            "Cache-Control": "no-store",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Keep JSON error bodies compatible with the original frontend ({error: ...})."""
    from fastapi.responses import JSONResponse

    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(status_code=exc.status_code, content={"error": detail}, headers=exc.headers)
