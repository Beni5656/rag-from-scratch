import wikipediaapi
from sentence_transformers import SentenceTransformer
import numpy as np
import anthropic
from dotenv import load_dotenv
import os

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

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def retrieve(query, chunks, chunk_embeddings, model, k=2):
    query_embedding = model.encode([query])[0]

    similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]

    top_k_indices = np.argsort(similarities)[::-1][:k]

    return [(chunks[i], similarities[i]) for i in top_k_indices]

query = "When did the rover land on Mars?"
results = retrieve(query, chunks, chunk_embeddings, model, k=2)

for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Chunk: {chunk[:200]}...")
    print()

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([chunk for chunk, score in retrieved_chunks])

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

query = "When did the rover land on Mars?"
retrieved = retrieve(query, chunks, chunk_embeddings, model, k=2)
answer = generate_answer(query, retrieved)

print(f"Question: {query}")
print(f"Answer: {answer}")