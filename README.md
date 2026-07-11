# 📄 AI PDF Assistant

A lightweight, ground-up **RAG (Retrieval-Augmented Generation)** assistant that processes PDF documents to answer questions, generate summaries, extract tables, discover dates, and produce reports.

This project is built from scratch without bulky orchestration frameworks (like LangChain or LlamaIndex) to showcase clean, modular, and highly testable RAG design patterns. It features a robust **FastAPI backend** and an intuitive **Streamlit frontend**.

---

## 🚀 Key Features

*   **Custom Sliding-Window Chunker:** Smart character-level chunking with configurable overlap to preserve text context across chunk boundaries and retain page number mapping.
*   **Automatic Page Citations:** Under the hood, chunks maintain parent page references, allowing the assistant to cite sources precisely (e.g., `[p.3]`) when answering questions.
*   **Targeted Retrieval & Document Isolation:** Supports indexing multiple files and isolating queries to a specific document ID or searching across all ingested documents.
*   **Built-in Analytical Tools:** Beyond general Q&A, the assistant features pre-configured templates for specific tasks:
    *   **Summarization:** Thematic summaries structured with bullet points.
    *   **Table Extraction:** Reconstructs tables from pages into clean GitHub-Flavored Markdown tables.
    *   **Timeline / Date Extraction:** Lists dates mentioned in the text and their significance.
    *   **Executive Report:** Generates structured reports containing Overview, Key Points, Risks, and Action Items.
*   **API & UI Separation:** Decoupled FastAPI server and Streamlit user interface, allowing independent scaling, deployment, or alternative client integrations.
*   **Test-Driven Design:** The pipeline depends on abstractions rather than concrete instances. This allows executing fast, dependency-free tests with zero API keys or heavy local models using in-memory fakes.

---

## 🛠️ Architecture Overview

The codebase is organized as follows:

```text
├── app/
│   └── streamlit_app.py       # Streamlit user interface
├── src/
│   └── pdf_assistant/
│       ├── __init__.py
│       ├── api.py             # FastAPI backend exposing RAG endpoints
│       ├── chunker.py         # Text chunking logic with overlap
│       ├── config.py          # Settings loader parsing .env file
│       ├── embeddings.py      # Embedding generation wrapper (Sentence-Transformers)
│       ├── errors.py          # Domain-specific custom exceptions
│       ├── extractor.py       # PDF parser using pypdf
│       ├── factory.py         # Factory to assemble & wire production components
│       ├── llm.py             # Anthropic Messages API client
│       ├── logging_config.py  # Standardized application logging
│       ├── pipeline.py        # Central orchestrator coordinating RAG steps
│       ├── prompts.py         # System and User prompts for tasks
│       └── vector_store.py    # ChromaDB persistent store wrapper
├── tests/                     # Test suite using lightweight in-memory mocks
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_config.py
│   ├── test_errors.py
│   └── test_pipeline.py
├── .env.example               # Template for environment configurations
├── pyproject.toml             # Project packaging metadata & dependencies
└── requirements.txt           # Lockfile for pip installations
```

---

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Mr-Gt-hash/ai-pdf-assistant.git
cd ai-pdf-assistant
```

### 2. Set Up a Virtual Environment
Create and activate a Python virtual environment (Python >= 3.9 is required):

```bash
# On Linux/macOS
python -m venv venv
source venv/bin/activate

# On Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install the package in editable mode along with its dependencies:

```bash
pip install -e .
```

*Alternatively, install directly from the requirements file:*
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

Open `.env` and add your **Anthropic API Key**:
```ini
# Required for Claude LLM integration
ANTHROPIC_API_KEY=your-sk-ant-key-here

# Optional overrides (defaults shown below)
LLM_MODEL=claude-opus-4-8
MAX_TOKENS=1024
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=4
PERSIST_DIR=./chroma_store
COLLECTION_NAME=pdf_documents
```

---

## 🚀 Running the Application

For the application to run, both the backend server and the frontend UI must be started.

### Step 1: Launch the FastAPI Backend
Start the backend server using Uvicorn. It will reload automatically when code changes are saved:

```bash
uvicorn pdf_assistant.api:app --reload
```
The API documentation and interactive swagger environment will be available at `http://localhost:8000/docs`.

### Step 2: Launch the Streamlit UI
In a separate terminal tab or window (with the virtual environment active), run:

```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser to interact with the application.

---

## 🔌 API Reference

| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | Returns backend operational status. |
| `/upload` | `POST` | Multipart File (`file`) | Ingests, chunks, embeds, and indexes a PDF. Returns a unique `doc_id`. |
| `/ask` | `POST` | `{"question": "...", "doc_id": "..."}` | Queries the document and returns a cited response. |
| `/summarize`| `POST` | `{"doc_id": "..."}` | Generates a structured summary grouped by themes. |
| `/tables` | `POST` | `{"doc_id": "..."}` | Identifies and reconstructs tables as Markdown. |
| `/dates` | `POST` | `{"doc_id": "..."}` | Lists all dates and events found in chronological context. |
| `/report` | `POST` | `{"doc_id": "..."}` | Produces a structured executive summary report. |

---

## 🧪 Running Tests

Tests run in complete isolation. Because the core pipeline is decoupled using protocols, the test suite injects lightweight fakes (`FakeEmbedder`, `FakeVectorStore`, `FakeLLM`) in `tests/conftest.py`. This ensures tests run instantly and locally without reading/writing files, loading heavy embedding models, or hitting external APIs.

To run the tests:
```bash
pytest
```

To run tests with code coverage metrics:
```bash
pytest --cov=src
```

---

## 📄 License

This project is licensed under the MIT License. See [pyproject.toml](file assistant/pyproject.toml) for details.
