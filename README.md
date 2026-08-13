# 📈 Equity RAG Agent

**A local-first, privacy-preserving financial research assistant.** Ask questions about corporate filings, shareholder letters, and trade logs — get answers grounded in your own documents, plus on-demand financial sentiment analysis. Everything runs on-device via Ollama; no API keys, no data leaves your machine.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-Agents-1C3C3C?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct%20Agent-0084FF" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-6C4EE3" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Overview

Equity RAG Agent is an agentic financial analyst that combines **retrieval-augmented generation** over your local documents with **domain-specific sentiment analysis** — all powered by open-source models running entirely on your own hardware. Point it at a folder of filings, ask it questions in plain English, and it decides on its own whether to search your documents, analyze sentiment, or answer directly.

## Features

- 🔍 **RAG-powered document search** — semantic search over local financial filings using ChromaDB and sentence-transformer embeddings
- 📊 **Financial sentiment analysis** — powered by [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned specifically on financial text (not generic sentiment)
- 🤖 **Agentic tool routing** — a local LLM autonomously decides when to retrieve documents vs. analyze sentiment vs. answer directly, built on [LangGraph](https://github.com/langchain-ai/langgraph)'s ReAct agent pattern
- 🔒 **100% local inference** — no OpenAI/Anthropic API keys, no data sent to third parties
- 💬 **Persistent chat UI** — clean Streamlit interface with conversation history

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) (`llama3.2:3b`) |
| Agent framework | LangGraph (`create_react_agent`) |
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sentiment model | FinBERT (`ProsusAI/finbert`) via 🤗 Transformers |
| UI | Streamlit |

## Architecture

```
User Query
    │
    ▼
Streamlit Chat UI
    │
    ▼
LangGraph ReAct Agent (llama3.2:3b)
    │
    ├──▶ retrieve_financial_docs()  ──▶ ChromaDB similarity search ──▶ local filings
    │
    └──▶ analyze_sentiment()        ──▶ FinBERT pipeline ──▶ Positive / Negative / Neutral
```

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~4 GB free RAM for the LLM, plus space for embedding/sentiment models

### 1. Install Ollama and pull the model

```bash
ollama pull llama3.2:3b
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/Equity-Rag-Agent.git
cd Equity-Rag-Agent
```

### 3. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Add your documents

Create a `data/` folder in the project root and place your financial documents inside (PDFs, CSVs, or text files):

```
data/
├── annual_report_2025.pdf
├── shareholder_letter.pdf
└── earnings_transcript.csv
```

> **Note:** The `data/` folder is intentionally excluded from version control (see `.gitignore`) — you supply your own documents locally.

### 5. Build the vector database

```bash
python ingest.py
```

This chunks and embeds your documents into a local ChromaDB store (`chroma_db/`), also excluded from git.

### 6. Run the app

```bash
ollama serve      # skip if Ollama is already running in the background
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Project Structure

```
.
├── app.py              # Streamlit UI + LangGraph agent definition
├── sentiment.py         # FinBERT sentiment analysis wrapper
├── ingest.py             # Document loader + ChromaDB vector store builder
├── requirements.txt
├── .gitignore
├── data/                  # Your source documents (not tracked in git)
└── chroma_db/             # Generated vector store (not tracked in git)
```

## Example Queries

```
"What did the company say about supply chain risks in the latest filing?"
"Analyze the sentiment of: revenue grew 20% but margins contracted due to rising input costs"
"Summarize the key risk factors mentioned in the shareholder letter"
```

## Roadmap

- [ ] Support for multi-turn conversational memory across sessions
- [ ] Source citation / page references in retrieved answers
- [ ] Support for larger local models (e.g. `qwen3:8b`) as an optional toggle
- [ ] Docker Compose setup for one-command local deployment

## Why Local-Only?

This project runs entirely on local inference (Ollama + local embeddings + local FinBERT) rather than cloud APIs, which means:

- Sensitive financial documents never leave your machine
- Zero per-query API costs
- Works fully offline once models are pulled

The trade-off: it isn't deployed to a public URL, since free cloud hosting tiers don't support persistent local LLM servers. To run it, clone the repo and follow the setup steps above.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Ollama](https://ollama.com) for local LLM serving
- [LangChain](https://www.langchain.com) / [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis
- [ChromaDB](https://www.trychroma.com) for vector storage
