import wikipediaapi
from sentence_transformers import SentenceTransformer

wiki = wikipediaapi.Wikipedia(user_agent="rag-learning-project", language="en")
page = wiki.page("Perseverance (rover)")

print(page.exists())
print(page.title)
print(len(page.text))
print(page.text[:500])

def chunk_text(text, min_length=100):
    paragraphs = text.split("\n\n")

    chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

    return chunks

chunks = chunk_text(page.text)

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[0]}")

model = SentenceTransformer('all-MiniLM-L6-v2')

chunk_embeddings = model.encode(chunks)

print(f"Number of embeddings: {len(chunk_embeddings)}")
print(f"Shape of one embedding: {chunk_embeddings[0].shape}")
print(f"First few numbers of first embedding: {chunk_embeddings[0][:5]}")