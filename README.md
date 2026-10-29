# Azure AI Generative Studio — Python API

FastAPI + Streamlit app using Azure OpenAI Responses API and the SmartCartAI e-commerce prompt.

## Features
- Chat completions (`POST /api/chat`)
- Uploadable PDF knowledge base with indexing, listing and deletion (`/api/rag/*`)
- Image analysis (`POST /api/vision`)
- Prompt-to-SVG illustration generation with PNG output (`POST /api/generate-image`)
- Health check (`GET /api/health`)

## PDF RAG configuration
Keep your existing `AZURE_ENDPOINT`, `AZURE_API_KEY`, and `AZURE_MODEL` values.

Add application settings:
- `RAG_ENABLED=true`
- `AZURE_EMBEDDING_MODEL=your-embedding-deployment-name` (a separate embedding deployment, e.g. `text-embedding-3-small`)
- `RAG_DATA_DIR=/home/data/smartcart-rag` (use a persistent writable directory/volume)
- Optional `RAG_MAX_PDF_BYTES=26214400` (25 MiB)
- Optional `RAG_TOP_K=4`
- Optional `AZURE_OPENAI_API_VERSION=2024-10-21` for standard Azure OpenAI embedding endpoints

The configured Foundry `/openai/v1/responses` URL is detected and LangChain's OpenAI-compatible embeddings client uses the `/openai/v1` base. Ensure the embedding deployment exists; GPT-4.1-mini is the chat model, not an embedding model.

## Upload workflow
1. Deploy the code and install `requirements.txt`.
2. Set `RAG_ENABLED`, `AZURE_EMBEDDING_MODEL`, and durable `RAG_DATA_DIR`.
3. Open Streamlit and select **Knowledge Base**.
4. Upload a PDF and click **Upload and index PDF**.
5. Ask a question in Chat; relevant chunks become reference context.
6. List or remove PDFs in the Knowledge Base tab; removal rebuilds the index.

Text-based PDFs are supported directly. Scanned/image-only PDFs need OCR. The upload/delete API is not authenticated in this initial version, so keep the application private or add authentication before public production use. Store data under persistent `RAG_DATA_DIR`; multiple replicas should use shared storage or a managed vector database.

## Local setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload --host 127.0.0.1 --port 3001
```
Run existing regression tests with `python -m unittest discover -s tests -v`.
