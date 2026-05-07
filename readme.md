# InquiryBot LangChain RAG

InquiryBot is a local Streamlit chatbot for asking questions over PDF documents. It
uses OpenAI embeddings, ChromaDB for persistent vector search, and LangChain for the
retrieval and answer-generation flow.

## What It Does

- Ingests PDFs from the `data/` folder or from the Streamlit sidebar uploader.
- Stores vectors locally in `chroma_db/`.
- Uses one shared Chroma collection and one embedding model everywhere.
- Retrieves relevant chunks with a similarity score threshold.
- Answers only from document context and politely refuses unsupported questions.
- Shows citations with document name, page number, relevance score, and chunk text.
- Can capture basic lead details and generate an email draft for business inquiries.
- Logs runtime errors to `logs/app.log`.

## Project Structure

```text
Inquirybot-langchain/
|-- chatbot.py             # Streamlit UI
|-- ingest_database.py     # CLI ingestion/rebuild script
|-- config.py              # Shared paths, models, and retrieval settings
|-- embeddings.py          # Shared OpenAI embedding client
|-- database.py            # Chroma, PDF loading, ingestion, retrieval helpers
|-- utils.py               # Logging, upload saving, citations, lead CSV helpers
|-- data/                  # PDF documents
|-- tests/
|   `-- test_retrieval.py  # Basic retrieval tests
|-- requirements.txt
|-- .env.example
`-- readme.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements-local.txt
```

Create a `.env` file:

```bash
copy .env.example .env
```

Then add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Configuration

Core settings live in `config.py`.

Current defaults:

- Chroma collection: `inquirybot_documents`
- Embedding model: `text-embedding-3-large`
- Default chat model: `gpt-4.1-mini`
- Alternative chat model: `gpt-4o-mini`
- Similarity threshold: `0.35`
- Retrieved chunks per query: `5`

## Ingest Documents

Place PDFs in `data/`, then run:

```bash
python ingest_database.py
```

To clear and rebuild the full vector database from all PDFs in `data/`:

```bash
python ingest_database.py --rebuild
```

You can also ingest specific PDFs:

```bash
python ingest_database.py "data/iSparx Business profile .pdf"
```

## Run the App

```bash
streamlit run chatbot.py
```

In the sidebar you can:

- Upload PDFs and ingest them immediately.
- Choose the OpenAI chat model.
- Ingest all PDFs from `data/`.
- Clear and rebuild the vector database.
- Capture lead details.
- Enable email draft generation.

## Citations

Each answer includes expandable source citations with:

- Document name
- Page number
- Relevance score
- Retrieved chunk text

If no retrieved chunk passes the similarity threshold, the app returns a refusal
message instead of guessing.

## Tests

Run:

```bash
python -m unittest discover tests
```

The tests use deterministic fake embeddings and a temporary Chroma database, so
they do not require OpenAI API calls or extra test dependencies.

## Notes

- `chroma_db/`, `.env`, `venv/`, logs, and generated lead CSV files are ignored by git.
- Uploaded PDFs are saved into `data/`.
- Lead capture is saved to `leads.csv`.

## Vercel

This repository includes `app.py` as a minimal WSGI entrypoint so Vercel's
Python builder can detect an application. Vercel installs from the minimal
`requirements.txt`, while the full local chatbot dependencies live in
`requirements-local.txt`.

The full chatbot UI is still the Streamlit app in `chatbot.py`, so run it with:

```bash
streamlit run chatbot.py
```

For a hosted interactive chatbot, prefer Streamlit Community Cloud, Render, or a
server/VM that can run a persistent Streamlit process.

See `DEPLOYMENT.md` for the Vercel Free bundle-size strategy.
