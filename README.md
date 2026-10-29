# Azure AI Generative Studio — Python API

FastAPI + Streamlit app using Azure OpenAI Responses API and the SmartCartAI e-commerce prompt.

## Features
- Chat completions (POST /api/chat)
- Uploadable PDF knowledge base with indexing, listing and deletion (/api/rag/*)
- Image analysis (POST /api/vision)
- Prompt-to-SVG illustration generation with PNG output (POST /api/generate-image)
- Health check (GET /api/health)

## PDF RAG without a separate embeddings deployment
Keep your existing AZURE_ENDPOINT, AZURE_API_KEY, and AZURE_MODEL settings unchanged. GPT-4.1-mini remains the only Azure model required.

The PDF knowledge base uses LangChain BM25 keyword retrieval, not vector embeddings. It extracts PDF text, splits it into overlapping chunks, stores those chunks as JSON, and ranks them against each chat question using BM25. This avoids deploying or paying for a separate embedding model. Keyword retrieval is less semantic than vector search, but requires no embedding endpoint.

Add these app variables:
- RAG_ENABLED=true
- RAG_DATA_DIR=/home/data/smartcart-rag (use a persistent writable directory or mounted volume)
- RAG_TOP_K=4 (optional; number of chunks to retrieve)
- RAG_MAX_PDF_BYTES=26214400 (optional; 25 MiB upload limit)

Do NOT set AZURE_EMBEDDING_MODEL; it is no longer used. RAG_DATA_DIR must persist across app restarts or the uploaded PDFs and chunk file can be lost.

## Upload workflow
1. Deploy the code and install requirements.txt.
2. Open the Streamlit app and select Knowledge Base.
3. Upload a PDF and click Upload and index PDF.
4. Ask questions in Chat; LangChain BM25 retrieves relevant passages and passes them to GPT-4.1-mini as untrusted reference context.
5. List or remove PDFs in the Knowledge Base tab; removal rebuilds the text-chunk index.

Text-based PDFs are supported directly. Scanned/image-only PDFs need OCR. The upload/delete API is not authenticated in this initial version, so keep the application private or add authentication before public production use. For multiple replicas, use shared durable storage.

## Local setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 3001
```
Run regression tests with `python -m unittest discover -s tests -v`.
