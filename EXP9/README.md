# Development of a Personalized AI Assistant using an Open-Source Large Language Model, RAG, and Fine-Tuning

This local college demonstration provides a browser chat interface for Qwen2.5-1.5B-Instruct and your existing LoRA adapter running in Google Colab. Flask keeps the Colab bearer token on the server. For document questions, the backend extracts PDF, DOCX, or TXT text, chunks it, embeds chunks locally with `all-MiniLM-L6-v2`, and sends the most relevant excerpts to the existing Colab `/generate` endpoint.

## Architecture

- `templates/index.html`, `static/style.css`, and `static/script.js` provide a responsive, framework-free chat interface.
- `app.py` handles chat, document uploads, extraction, local retrieval, and authenticated requests to Colab.
- Chat history and preferences are stored in this browser's `localStorage`.
- The sentence-transformers embedding model is downloaded and run locally on first document processing, then cached for the lifetime of the Flask process. The Qwen model and LoRA adapter are not loaded locally.

## Prerequisites

- Python 3.10 or newer.
- Your Colab notebook must load the existing Qwen model and LoRA adapter, start the Flask inference API, and expose it using a Cloudflare Quick Tunnel.
- Internet access is needed for the temporary tunnel and for the first download of the local embedding model. The embedding model is separate from the Qwen generation model.

## Install

From the project folder, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch, required by sentence-transformers, is a sizeable dependency. If installation fails on your machine, install a compatible PyTorch build for your Python/macOS version first using the official PyTorch installation selector, then rerun `python -m pip install -r requirements.txt`.

## Configure Colab

Create a `.env` file in this project folder (do not commit or share it):

```dotenv
COLAB_API_URL=https://your-current-cloudflare-quick-tunnel.trycloudflare.com
COLAB_API_TOKEN=your_existing_bearer_token
```

Use the tunnel's HTTPS base URL only, without `/health` or `/generate`. Keep your existing token private. Flask reads both values from the environment and never sends them to the browser.

In Colab, keep the notebook cell that loads `Qwen/Qwen2.5-1.5B-Instruct` and your fine-tuned LoRA adapter running. Start the authenticated Flask API with its `GET /health` and `POST /generate` routes, then start the Cloudflare Quick Tunnel and copy its current HTTPS URL into `.env`. The adapter remains in Colab: this app does not load, replace, or retrain the language model.

## Run locally

With the virtual environment active and `.env` configured:

```bash
python app.py
```

Open `http://127.0.0.1:5000`. The app binds to localhost and Flask debug mode is disabled. To check local configuration and the authenticated Colab health endpoint, run:

```bash
curl -i http://127.0.0.1:5000/api/health
```

A reachable Colab endpoint returns `200` with `{"status":"ok","configured":true,"colab":"connected"}`. If it is disconnected or not configured, the endpoint returns a degraded status without exposing the URL or token. The page also displays connection status.

## Using documents

Upload up to 10 PDF, DOCX, or UTF-8 TXT files, each no larger than 10 MB. Uploaded documents are selected automatically; use the sidebar checkbox to choose which documents are sent for retrieval. The backend splits text into overlapping chunks and retrieves up to four chunks whose cosine similarity meets the default `0.30` threshold. The UI only displays document attribution when relevant chunks were actually retrieved. Questions without selected documents go to Colab with conversation and preference context.

Optional local retrieval settings can be set in `.env` before starting Flask:

```dotenv
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_SIMILARITY_THRESHOLD=0.30
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

Keep chunk overlap smaller than chunk size. Adjust the threshold based on your documents and evaluation results.

## Demo limitations and safety

- Cloudflare Quick Tunnel URLs are temporary and change when the tunnel restarts. Update `.env` and restart Flask when that happens. Colab disconnections, runtime resets, and model startup time interrupt generation; the Colab runtime and tunnel must stay active while using the app.
- Uploaded file records and embeddings live in Flask process memory. Restarting Flask clears the document list and retrieval index; files left in `uploads/` are not automatically re-indexed. Removing a listed document deletes its generated-ID file. This is intentionally a single-process college demo, not durable storage or a production deployment.
- Chat history and preferences live in the current browser's local storage. Clearing browser storage removes them. Do not use this local demo for confidential documents or expose its Flask port publicly.
- Text extraction cannot read scanned/image-only PDFs without OCR. Malformed or encrypted files may fail. Uploads are limited to PDF, DOCX, and UTF-8 TXT, use generated storage names, and are size limited.
- The local embedding model may download from Hugging Face the first time a document is processed. If extraction or embedding fails, the request reports an error instead of pretending retrieval succeeded.

## Project files

```text
app.py
requirements.txt
README.md
.gitignore
templates/index.html
static/style.css
static/script.js
uploads/.gitkeep
```
