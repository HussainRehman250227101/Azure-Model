"""Streamlit frontend for the Azure AI Generative Studio FastAPI backend.

Run with: streamlit run streamlit_app.py
The Azure API key stays in the FastAPI backend's .env file.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="Azure AI Generative Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_BASE = os.getenv("STREAMLIT_API_BASE", "http://127.0.0.1:3001").strip().rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(95.0, connect=8.0)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 18px; margin-bottom: 1.3rem;
             background: linear-gradient(115deg, rgba(59,130,246,.11), rgba(139,92,246,.08));}
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: .55rem 0 0 0; color: var(--text-color); opacity: .78;}
      div[data-testid="stChatMessage"] {border-radius: 14px;}
      .small-muted {font-size: .86rem; opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>✨ Azure AI Generative Studio</h1>
      <p>Chat with your model, describe images, and create downloadable vector-style PNG artwork.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return f"Request failed ({response.status_code})."
    if isinstance(data, dict):
        return str(data.get("error") or data.get("detail") or f"Request failed ({response.status_code}).")
    return f"Request failed ({response.status_code})."


def get_health(base_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(api_url(base_url, "/api/health"), timeout=5.0)
        if response.is_success:
            data = response.json()
            return True, f"Connected · model: {data.get('model', 'not configured')}"
        return False, f"Backend responded with HTTP {response.status_code}."
    except httpx.HTTPError:
        return False, "Cannot reach the API. Start the FastAPI backend first."


with st.sidebar:
    st.subheader("⚙️ Connection")
    base_url = st.text_input(
        "FastAPI backend URL",
        value=st.session_state.get("api_base_url", DEFAULT_API_BASE),
        help="The Python backend from this repository. Default: http://127.0.0.1:3001",
    ).strip().rstrip("/")
    st.session_state["api_base_url"] = base_url or DEFAULT_API_BASE
    if st.button("Check connection", use_container_width=True):
        st.session_state["check_connection_now"] = True
    connected, health_text = get_health(st.session_state["api_base_url"])
    if connected:
        st.success(health_text)
    else:
        st.warning(health_text)
    st.divider()
    st.caption("Azure credentials remain in the backend .env file. Do not add your API key here.")
    st.caption("Image uploads: PNG, JPG/JPEG, WebP · Maximum 10 MiB")


api_base = st.session_state["api_base_url"]

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Azure AI assistant. Ask me a question or try one of the studio tools.",
        }
    ]

chat_tab, analyze_tab, generate_tab = st.tabs(
    ["💬 Chat", "🔎 Analyze image", "🎨 Generate PNG"]
)

with chat_tab:
    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.subheader("Chat with Azure AI")
    with clear_col:
        if st.button("New chat", use_container_width=True):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "New conversation started. What would you like to explore?",
                }
            ]
            st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Message your AI assistant...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        # Keep the sent history within the same bounds as the backend.
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.chat_messages
            if item["role"] in {"user", "assistant"}
        ][-20:]
        try:
            with st.spinner("Azure AI is thinking..."):
                response = httpx.post(
                    api_url(api_base, "/api/chat"),
                    json={"messages": history},
                    timeout=REQUEST_TIMEOUT,
                )
            if not response.is_success:
                raise RuntimeError(error_message(response))
            result: dict[str, Any] = response.json()
            answer = result.get("reply")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The API returned an empty response.")
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer.strip()}
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": f"**Request failed:** {exc}",
                }
            )
        st.rerun()

with analyze_tab:
    st.subheader("Analyze an image")
    st.write("Upload a picture and Azure AI will describe its main visual details.")
    uploaded = st.file_uploader(
        "Choose an image",
        type=ALLOWED_IMAGE_TYPES,
        key="vision_upload",
        help="Supported formats: PNG, JPG/JPEG, and WebP. Maximum size is 10 MiB.",
    )
    if uploaded is not None:
        upload_bytes = uploaded.getvalue()
        if len(upload_bytes) > MAX_IMAGE_BYTES:
            st.error("Please choose an image that is 10 MiB or smaller.")
        else:
            st.image(upload_bytes, caption=uploaded.name, use_container_width=True)
            if st.button("Analyze image", type="primary", key="analyze_button"):
                try:
                    with st.spinner("Analyzing the image..."):
                        response = httpx.post(
                            api_url(api_base, "/api/vision"),
                            files={
                                "image": (
                                    uploaded.name,
                                    upload_bytes,
                                    uploaded.type or "application/octet-stream",
                                )
                            },
                            timeout=REQUEST_TIMEOUT,
                        )
                    if not response.is_success:
                        raise RuntimeError(error_message(response))
                    result = response.json()
                    description = result.get("reply", "No description was returned.")
                    st.session_state["last_image_analysis"] = description
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    st.error(f"Image analysis failed: {exc}")
    if st.session_state.get("last_image_analysis"):
        st.markdown("#### Image description")
        st.write(st.session_state["last_image_analysis"])

with generate_tab:
    st.subheader("Generate vector-style artwork")
    st.write(
        "Describe an illustration. The backend asks your Azure model to create SVG markup "
        "and converts it to a downloadable PNG."
    )
    with st.form("generate_image_form"):
        image_prompt = st.text_area(
            "Image description",
            placeholder="Example: A blue sports car on a modern city street at sunset, clean vector illustration",
            max_chars=2000,
            height=120,
        )
        st.caption(f"Maximum 2,000 characters.")
        submitted = st.form_submit_button("Generate PNG", type="primary")
    if submitted:
        if not image_prompt.strip():
            st.error("Please describe the image you want to create.")
        elif len(image_prompt.strip()) > 2000:
            st.error("Please keep the image description under 2,000 characters.")
        else:
            try:
                with st.spinner("Generating artwork and converting it to PNG..."):
                    response = httpx.post(
                        api_url(api_base, "/api/generate-image"),
                        json={"prompt": image_prompt.strip()},
                        timeout=REQUEST_TIMEOUT,
                    )
                if not response.is_success:
                    raise RuntimeError(error_message(response))
                content_type = response.headers.get("content-type", "")
                if "image/png" not in content_type.lower():
                    raise RuntimeError("The backend did not return a PNG image.")
                st.session_state["generated_png"] = response.content
                st.session_state["generated_png_prompt"] = image_prompt.strip()
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                st.error(f"Image generation failed: {exc}")

    generated_png = st.session_state.get("generated_png")
    if generated_png:
        st.markdown("#### Generated image")
        st.image(
            generated_png,
            caption=st.session_state.get("generated_png_prompt", "Generated artwork"),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download PNG",
            data=generated_png,
            file_name="generated-image.png",
            mime="image/png",
            type="primary",
        )

st.divider()
st.markdown(
    '<p class="small-muted">Powered by your Azure deployment · Generated outputs may need review.</p>',
    unsafe_allow_html=True,
)
