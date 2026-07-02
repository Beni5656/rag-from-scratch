import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent="rag-learning-project", language="en")
page = wiki.page("Perseverance (rover)")

print(page.exists())
print(page.title)
print(len(page.text))
print(page.text[:500])

