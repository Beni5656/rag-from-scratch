import wikipediaapi

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

