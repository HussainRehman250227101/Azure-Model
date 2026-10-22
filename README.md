# Azure AI Generative Studio — Python API

Python/FastAPI implementation of the backend in [Azure-Models](https://github.com/HussainRehman250227101/Azure-Models).

## Features

- Chat completions using conversation history (`POST /api/chat`)
- Image analysis for PNG, JPG/JPEG, and WebP (`POST /api/vision`)
- Prompt-to-SVG illustration generation with PNG output (`POST /api/generate-image`)
- Health check (`GET /api/health`)

The API key stays on the server. Do not put Azure credentials in frontend code or commit your `.env` file.

## Requirements

- Python 3.10 or newer
- An Azure OpenAI resource and a deployment that supports the Responses API and image input

## Local setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `AZURE_ENDPOINT`: your Azure OpenAI resource endpoint. The code accepts a base endpoint, an endpoint ending in `/openai/v1`, or the full `/openai/v1/responses` URL.
- `AZURE_API_KEY`: key from your Azure resource.
- `AZURE_MODEL`: the **deployment name** to use for the Responses API.
- `PORT` (optional; defaults to `3001`).
- `CORS_ORIGINS` (optional; comma-separated browser origins, defaults to Vite localhost origins).

Run the backend:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 3001
```

API docs are available at `http://127.0.0.1:3001/docs`.

## Fixing SVG-to-PNG conversion errors

If you pulled an older version of this project and see `Could not convert the generated SVG to PNG`, stop the backend and update the dependencies/code:

```powershell
python -m pip install --upgrade -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 3001
```

The current backend uses `resvg-py` instead of CairoSVG to avoid missing Cairo native libraries, and logs renderer details to the backend terminal if the model returns invalid SVG.

## API examples

### Health

```powershell
Invoke-RestMethod http://127.0.0.1:3001/api/health
```

### Chat

```powershell
$body = @{ messages = @(@{ role = 'user'; content = 'Explain machine learning simply.' }) } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:3001/api/chat -Method Post -ContentType 'application/json' -Body $body
```

### Generate an image

```powershell
$body = @{ prompt = 'A blue sports car in a clean vector illustration style' } | ConvertTo-Json
Invoke-WebRequest http://127.0.0.1:3001/api/generate-image -Method Post -ContentType 'application/json' -Body $body -OutFile generated.png
```

### Analyze an image

```powershell
curl.exe -X POST http://127.0.0.1:3001/api/vision -F "image=@C:\path\to\image.png"
```

## Frontend integration

The original frontend defaults to `http://localhost:3001`, which matches this backend. Start it with the existing React/Vite app, then run this API separately. If your frontend uses another port, add that exact origin to `CORS_ORIGINS`.

If your local frontend is still part of the `Azure-Models` repository, keep the React/Vite project there and run this Python API from the `Azure` folder. The API contract is kept compatible with the frontend (`reply` JSON values, `{error: ...}` failures, and binary PNG output).

## Notes

- Image upload limit is 10 MiB; accepted formats are PNG, JPEG, and WebP.
- Generated art is SVG authored by the text model and rasterized to PNG using `resvg-py`, which includes a prebuilt renderer and does not require native Cairo/GTK libraries on Windows. It is vector-style illustration, not native photorealistic image generation.
- Azure deployment/API support varies by resource and model. If a request fails, the API returns Azure's error message when available.
