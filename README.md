# RAG From Scratch

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python — no LangChain, no LlamaIndex — to understand exactly how chunking, embedding, retrieval, and generation work under the hood.

## What it does

Given a Wikipedia article, this pipeline:
1. Fetches the raw article text
2. Splits it into clean, meaningful chunks
3. Converts each chunk into a vector embedding
4. Retrieves the most relevant chunks for a given question using cosine similarity
5. Passes those chunks to an LLM to generate a grounded answer

## Pipeline

| Step | What happens | Tools used |
|------|---------------|------------|
| Fetch | Pull article text from Wikipedia | `wikipedia-api` |
| Chunk | Split text into paragraphs, filter out junk | Python string ops |
| Embed | Convert each chunk into a 384-dim vector | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Retrieve | Compare query vs. chunk vectors via cosine similarity | `numpy` |
| Generate | Answer the question using only retrieved context | Anthropic API (Claude) |

## Setup

```bash
git clone <https://github.com/Beni5656/rag-from-scratch>
cd rag-from-scratch

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root: