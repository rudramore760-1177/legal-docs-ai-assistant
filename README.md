# ⚖️ Legal Document Q&A Assistant

An AI-powered **RAG (Retrieval-Augmented Generation)** application that answers natural-language questions about legal documents — contracts, NDAs, and agreements — with answers grounded strictly in the uploaded document text, not model guesswork.

🔗 **Live demo:** [your-streamlit-link-here.streamlit.app](#)
📂 **Source code:** you're looking at it

![App screenshot](screenshot.png)

---

## Overview

Legal documents are dense, long, and hard to search manually. This project lets a user upload contracts or agreements and ask plain-English questions — *"What's the security deposit?"*, *"Is there a non-compete clause?"* — and get an answer sourced directly from the relevant clause, with the exact retrieved text shown for verification.

The system is built as a full retrieval pipeline rather than a single API call: documents are chunked, embedded, indexed, and only the most relevant sections are passed to the language model at query time — which keeps answers accurate and auditable instead of relying on the model's general knowledge.

## Features

- 📄 **Multi-document upload** — index several contracts at once and query across all of them
- 🔍 **Retrieval-grounded answers** — every response is generated only from retrieved document chunks, with an explicit instruction to say "not found" rather than guess
- 📎 **Source transparency** — each answer includes an expandable panel showing the exact retrieved text it was built from
- 🚩 **"Not found" flagging** — answers where the system couldn't locate relevant information are visually flagged, rather than silently blending in with confident answers
- ⚡ **Response time + retrieval metrics** — every answer displays how long it took and how many chunks were retrieved
- 📊 **Session dashboard** — tracks total questions asked and average response time
- ⬇️ **Exportable session reports** — download a full Q&A transcript as a text file
- 💡 **Contextual follow-up suggestions** — the app suggests a relevant next question based on what was just asked
- 🎨 **Custom dark UI** — a bold, distinct interface rather than default Streamlit styling

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | [LlamaIndex](https://www.llamaindex.ai/) | Purpose-built for indexing and querying private document collections, rather than general-purpose agent tooling |
| Vector store | [ChromaDB](https://www.trychroma.com/) | Lightweight, free, runs locally with no external dependency |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` | Free, runs locally, no per-query cost or API dependency |
| LLM | Google Gemini (`gemini-3.6-flash`) | Free tier, strong reasoning quality, no billing setup required |
| Frontend | [Streamlit](https://streamlit.io/) | Fast to build a real, deployable chat UI without a separate frontend framework |

## Architecture

```
 Upload PDF(s)
      │
      ▼
 Document parsing & chunking  (LlamaIndex)
      │
      ▼
 Chunk embedding  (HuggingFace, local)
      │
      ▼
 Vector storage  (ChromaDB)
      │
      ▼
 User asks a question
      │
      ▼
 Similarity search → top-k relevant chunks retrieved
      │
      ▼
 Chunks + question → Gemini  (generation, grounded in retrieved text only)
      │
      ▼
 Answer + source citations shown to user
```

## Design decisions worth noting

- **Free, local embeddings over a paid API** — embeddings are generated on-device using an open-source model, removing per-query cost and external dependency for that stage of the pipeline. Generation still uses a hosted LLM (Gemini), since that reasoning step benefits most from a frontier-quality model.
- **Explicit grounding instruction** — the system prompt directly instructs the model to decline rather than fabricate an answer when the documents don't contain the requested information. This is tested against out-of-scope questions (e.g., asking about salary terms in a rental agreement) to confirm the system doesn't hallucinate across documents.
- **Source citation by default** — every answer surfaces its retrieved chunks, so correctness can be manually verified rather than trusted blindly — an important property for anything positioned as a "legal" tool, even a demo one.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rudramore760-1177/legal-docs-ai-assistant.git
cd legal-docs-ai-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API key:
   ```bash
   cp .env.example .env
   ```
   Add a free Gemini API key (get one at [aistudio.google.com](https://aistudio.google.com), no credit card required) to `.env`.

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Sample documents

The `data/` folder includes three fictional legal documents for testing: a Non-Disclosure Agreement, a residential Rental Agreement, and an Employment Contract — each with realistic clauses covering termination, payment, confidentiality, and governing law.

Example questions to try:
- *"What is the security deposit for the rental agreement?"*
- *"How long is the non-compete clause in the employment contract?"*
- *"What is the notice period to terminate the NDA?"*

## Roadmap

- [ ] Automated evaluation pipeline using [Ragas](https://github.com/explodinggradients/ragas) (faithfulness, answer relevancy, context precision)
- [ ] Multi-turn conversational memory for follow-up questions
- [ ] Support for scanned/OCR'd PDFs and DOCX files
- [ ] Optional password-gated access for shared deployment
