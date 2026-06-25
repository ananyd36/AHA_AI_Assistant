# AHA! Chatbot

An AI-powered curriculum support tool for teachers delivering Edge AI lessons. Provides instant, step-by-step answers about Arduino IDE, ESP microcontrollers, Edge Impulse, and the 5 AHA curriculum modules.

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  Chrome Extension   │     │   Streamlit Web App  │
│   (aha_chatbot/)    │     │     (app/)           │
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
         ┌──────────────────────┐
         │   FastAPI Backend    │  ← Deployed on Fly.io
         │   (langchain_crash/) │    https://langchain-crash.fly.dev
         └──────────┬───────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   ┌─────────────┐    ┌───────────────┐
   │  Pinecone   │    │  SQLite (DB)  │
   │ Vector Store│    │  (Fly Volume) │
   └─────────────┘    └───────────────┘
```

**Both interfaces share the same backend.** Improvements to the knowledge base or model quality apply everywhere simultaneously.

---

## Components

### Backend — `langchain_crash/`

FastAPI server handling chat, document ingestion, and retrieval.

| File | Purpose |
|------|---------|
| `main.py` | API endpoints (`/chat`, `/upload-doc`, `/delete-doc`, `/list-docs`, `/health`) |
| `langchain_utils.py` | RAG pipeline: query routing, multi-query retrieval, history-aware chain |
| `pinecone_utils.py` | Pinecone indexing: contextual chunking, hybrid BM25 + dense upsert |
| `db_utils.py` | SQLite helpers for chat history and document records |
| `pydantic_models.py` | Request/response models |
| `ingest.py` | One-time bulk ingestion script for local PDFs in `./data/` |

### Chrome Extension — `aha_chatbot/`

Side-panel extension (Manifest V3) with chat and document management tabs. Submitted to Chrome Web Store — distributed to trusted users only.

| File | Purpose |
|------|---------|
| `sidebar.html` | Side panel UI |
| `sidebar.js` | API calls, chat persistence via `chrome.storage.local`, markdown rendering |
| `background.js` | Opens side panel on icon click |
| `styles.css` | Dark theme styling |
| `marked.min.js` | Local markdown parser |

### Web App — `app/`

Streamlit interface for team testing and iteration. Deployed on Streamlit Community Cloud.

---

## RAG Pipeline

```
User query
  → Query router (greeting / out_of_scope / rag)
  → History-aware reformulation (standalone question from chat history)
  → MultiQueryRetriever
      ├─ Generates 3 curriculum-scoped query variants
      ├─ Pinecone hybrid search (BM25 + dense, top_k=15 per variant)
      └─ Deduplicates → up to ~35 unique chunks
  → gpt-4o-mini with system prompt + context
  → Answer + top 3 unique source documents
```

**Ingestion pipeline** (per document):
1. PDF → chunks (4000 chars, 500 overlap)
2. LLM generates 1–2 sentence context per chunk relative to the full document
3. Context prepended to chunk text
4. BM25 sparse + OpenAI dense vectors upserted to Pinecone

---

## Environment Variables

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
```

Pinecone index must be configured with:
- **Dimensions**: 256
- **Metric**: dotproduct (required for hybrid search)

---

## Running Locally

**Backend**
```bash
cd langchain_crash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
uvicorn main:app --reload
```

**Streamlit app**
```bash
cd langchain_crash/app
pip install streamlit requests
streamlit run streamlit_app.py
```

**Bulk document ingestion**
```bash
# Place PDFs in langchain_crash/data/
python ingest.py
```

---

## Deployment

### Backend (Fly.io)

Live at: `https://langchain-crash.fly.dev`

```bash
fly deploy --ha=false
```

Auto-deploys on push to `main` via GitHub Actions (skips deploy when only `app/` changes).

Persistent volume `aha_data` mounted at `/data` keeps SQLite between redeploys.

### Streamlit App (Streamlit Community Cloud)

Deploy at [share.streamlit.io](https://share.streamlit.io):
- Repository: this repo
- Main file: `app/streamlit_app.py`

### Chrome Extension

Submitted to Chrome Web Store (version 1.1). Distributed to trusted users only pending review approval.

To load locally for testing:
1. `chrome://extensions` → Enable Developer Mode
2. Load Unpacked → select `aha_chatbot/`

---

## Data & Access

- **Knowledge base**: curriculum PDFs are private; ingested into Pinecone and not exposed publicly
- **Extension**: restricted distribution via Chrome Web Store — approved users only
- **Web app**: for internal team testing
