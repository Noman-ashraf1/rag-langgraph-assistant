from db.vectorstore import search

docs = search("what is AI", k=3)

print("\n🔴 RAW DOCS:")
for d in docs:
    print(d)