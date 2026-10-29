"""Streamlit UI for SmartCartAI with PDF upload and management."""
from __future__ import annotations
import os
from typing import Any
import httpx
import streamlit as st
st.set_page_config(page_title="SmartCartAI",page_icon="✨",layout="wide",initial_sidebar_state="expanded")
DEFAULT_API_BASE=os.getenv("STREAMLIT_API_BASE","http://127.0.0.1:3001").strip().rstrip("/")
REQUEST_TIMEOUT=httpx.Timeout(180.0,connect=8.0)
MAX_IMAGE_BYTES=10*1024*1024
MAX_PDF_BYTES=25*1024*1024
ALLOWED_IMAGE_TYPES=["png","jpg","jpeg","webp"]
def api_url(base_url:str,path:str)->str: return f"{base_url.rstrip(chr(47))}/{path.lstrip(chr(47))}"
def error_message(response:httpx.Response)->str:
    try: data=response.json()
    except ValueError: return f"Request failed ({response.status_code})."
    return str(data.get("error") or data.get("detail") or f"Request failed ({response.status_code}).") if isinstance(data,dict) else f"Request failed ({response.status_code})."
def get_health(base_url:str)->tuple[bool,str]:
    try:
        r=httpx.get(api_url(base_url,"/api/health"),timeout=5.0)
        if r.is_success:
            d=r.json()
            return True,f"Connected · model: {d.get('model','not configured')} · {int(d.get('indexed_pdf_count',0))} indexed PDF(s)"
        return False,f"Backend responded with HTTP {r.status_code}."
    except httpx.HTTPError: return False,"Cannot reach the API. Start the FastAPI backend first."
st.markdown("""<style>.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}.hero{padding:1.4rem 1.6rem;border:1px solid rgba(128,128,128,.25);border-radius:18px;margin-bottom:1.3rem;background:linear-gradient(115deg,rgba(59,130,246,.11),rgba(139,92,246,.08))}.hero h1{margin:0;font-size:2rem}.hero p{margin:.55rem 0 0;opacity:.78}.small-muted{font-size:.86rem;opacity:.72}</style>""",unsafe_allow_html=True)
st.markdown("""<div class="hero"><h1>🛒 SmartCartAI E-commerce Assistant</h1><p>Upload product catalogs and store documents, then ask questions grounded in your PDFs.</p></div>""",unsafe_allow_html=True)
with st.sidebar:
    st.subheader("⚙️ Connection")
    base_url=st.text_input("FastAPI backend URL",value=st.session_state.get("api_base_url",DEFAULT_API_BASE)).strip().rstrip("/")
    st.session_state["api_base_url"]=base_url or DEFAULT_API_BASE
    if st.button("Check connection",use_container_width=True): st.session_state["check_connection_now"]=True
    ok,health=get_health(st.session_state["api_base_url"])
    st.success(health) if ok else st.warning(health)
    st.divider()
    st.caption("Azure credentials stay in backend environment settings.")
    st.caption("PDF uploads: text-based PDFs, up to 25 MiB.")
api_base=st.session_state["api_base_url"]
if "chat_messages" not in st.session_state: st.session_state.chat_messages=[{"role":"assistant","content":"Hello! I am SmartCartAI. Upload a product catalog, manual or store policy in Knowledge Base, then ask questions about it here."}]
chat_tab,knowledge_tab,analyze_tab,generate_tab=st.tabs(["💬 Chat","📚 Knowledge Base","🔎 Analyze image","🎨 Generate PNG"])
with chat_tab:
    left,right=st.columns([5,1]); left.subheader("Chat with SmartCartAI")
    if right.button("New chat",use_container_width=True): st.session_state.chat_messages=[{"role":"assistant","content":"New conversation started. Ask about your documents or shopping."}]; st.rerun()
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    prompt=st.chat_input("Ask about your uploaded PDFs or shopping...")
    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        history=[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_messages if m["role"] in {"user","assistant"}][-20:]
        try:
            with st.spinner("Retrieving PDF context and asking Azure AI..."): r=httpx.post(api_url(api_base,"/api/chat"),json={"messages":history},timeout=REQUEST_TIMEOUT)
            if not r.is_success: raise RuntimeError(error_message(r))
            reply=r.json().get("reply")
            if not isinstance(reply,str) or not reply.strip(): raise RuntimeError("The API returned an empty response.")
            st.session_state.chat_messages.append({"role":"assistant","content":reply.strip()})
        except (httpx.HTTPError,RuntimeError,ValueError) as e: st.session_state.chat_messages.append({'role':'assistant','content':f'**Request failed:** {e}'})
        st.rerun()
with knowledge_tab:
    st.subheader("Upload PDF knowledge")
    st.write("PDFs are extracted, split into overlapping chunks, embedded, and stored in a searchable vector index.")
    pdf=st.file_uploader("Choose a PDF document",type=["pdf"],key="rag_pdf_upload",help="Max 25 MiB. Scanned/image-only PDFs need OCR first.")
    if pdf is not None:
        data=pdf.getvalue()
        st.caption(f"{pdf.name} · {len(data):,} bytes")
        if len(data)>MAX_PDF_BYTES: st.error("The PDF exceeds the 25 MiB limit.")
        elif not data.startswith(b"%PDF-"): st.error("The uploaded file is not a valid PDF.")
        elif st.button("Upload and index PDF",type="primary",key="rag_upload_button"):
            try:
                with st.spinner("Uploading, extracting text, and creating embeddings..."): r=httpx.post(api_url(api_base,"/api/rag/upload"),files={"file":(pdf.name,data,"application/pdf")},timeout=REQUEST_TIMEOUT)
                if not r.is_success: raise RuntimeError(error_message(r))
                out=r.json(); st.success(out.get("message","PDF indexed.")); st.caption(f"Text chunks indexed: {out.get('chunk_count',0)}"); st.rerun()
            except (httpx.HTTPError,RuntimeError,ValueError) as e: st.error(f"PDF upload failed: {e}")
    st.markdown("#### Indexed documents")
    try:
        r=httpx.get(api_url(api_base,"/api/rag/documents"),timeout=10)
        if not r.is_success: st.warning(error_message(r))
        else:
            docs=r.json().get("documents",[])
            if not docs: st.info("No PDFs uploaded yet.")
            for d in docs:
                c1,c2=st.columns([5,1]); c1.write(f"**{d.get('file_name','PDF')}** · {int(d.get('size_bytes',0))/(1024*1024):.2f} MiB"); c1.caption(f"SHA-256: {d.get('sha256','')}")
                if c2.button("Remove",key="remove_"+d.get("file_name","")):
                    try:
                        removed=httpx.delete(api_url(api_base,"/api/rag/documents/"+d.get("file_name","")),timeout=REQUEST_TIMEOUT)
                        if not removed.is_success: raise RuntimeError(error_message(removed))
                        st.success(removed.json().get("message","PDF removed.")); st.rerun()
                    except (httpx.HTTPError,RuntimeError,ValueError) as e: st.error(f"Could not remove PDF: {e}")
    except httpx.HTTPError: st.warning("Could not load documents. Check the backend connection.")
with analyze_tab:
    st.subheader("Analyze an image")
    image=st.file_uploader("Choose an image",type=ALLOWED_IMAGE_TYPES,key="vision_upload",help="PNG/JPG/WebP, max 10 MiB.")
    if image is not None:
        raw=image.getvalue()
        if len(raw)>MAX_IMAGE_BYTES: st.error("Image must be 10 MiB or smaller.")
        else:
            st.image(raw,caption=image.name,use_container_width=True)
            if st.button("Analyze image",type="primary",key="analyze_button"):
                try:
                    with st.spinner("Analyzing image..."): r=httpx.post(api_url(api_base,"/api/vision"),files={"image":(image.name,raw,image.type or "application/octet-stream")},timeout=REQUEST_TIMEOUT)
                    if not r.is_success: raise RuntimeError(error_message(r))
                    st.session_state["last_image_analysis"]=r.json().get("reply","No description returned.")
                except (httpx.HTTPError,RuntimeError,ValueError) as e: st.error(f"Image analysis failed: {e}")
    if st.session_state.get("last_image_analysis"): st.write(st.session_state["last_image_analysis"])
with generate_tab:
    st.subheader("Generate vector-style artwork")
    with st.form("generate_image_form"):
        description=st.text_area("Image description",max_chars=2000,height=120,placeholder="Describe the illustration...")
        submitted=st.form_submit_button("Generate PNG",type="primary")
    if submitted:
        if not description.strip(): st.error("Please describe the image.")
        else:
            try:
                with st.spinner("Generating and converting artwork..."): r=httpx.post(api_url(api_base,"/api/generate-image"),json={"prompt":description.strip()},timeout=REQUEST_TIMEOUT)
                if not r.is_success: raise RuntimeError(error_message(r))
                if "image/png" not in r.headers.get("content-type","").lower(): raise RuntimeError("Backend did not return a PNG.")
                st.session_state["generated_png"]=r.content; st.session_state["generated_png_prompt"]=description.strip()
            except (httpx.HTTPError,RuntimeError,ValueError) as e: st.error(f"Image generation failed: {e}")
    if st.session_state.get("generated_png"):
        st.image(st.session_state["generated_png"],caption=st.session_state.get("generated_png_prompt","Generated art"),use_container_width=True)
        st.download_button("⬇️ Download PNG",data=st.session_state["generated_png"],file_name="generated-image.png",mime="image/png")
st.divider()
st.markdown('<p class="small-muted">Powered by your Azure deployment · Generated outputs may need review.</p>',unsafe_allow_html=True)
