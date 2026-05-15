import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
INDEX_PATH = "db/faiss.index"
DOCS_PATH = "db/docs.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
docs = None


# =========================
# LOAD STORE (FAST PATH)
# =========================
def load_store():
    global index, docs

    if index is not None:
        return

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("📦 Loading FAISS index from disk...")

        index = faiss.read_index(INDEX_PATH)

        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)

        print("✅ Vectorstore loaded")
        return

    raise Exception("❌ No FAISS index found. Run build_vectorstore() first.")


# =========================
# BUILD VECTORSTORE (RUN ONCE)
# =========================
def build_vectorstore(chunks):
    global index, docs

    print("🔨 Building FAISS embeddings...")

    docs = chunks

    # convert dict → clean text
    texts = [
        f"{c.get('title','')}\n{c.get('text','')}"
        for c in chunks
    ]

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # cosine similarity normalization
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    print("📦 Adding vectors to FAISS...")
    index.add(embeddings)

    # save to disk
    os.makedirs("db", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("✅ Vectorstore saved successfully")


# =========================
# SEARCH FUNCTION
def search(query, k=5):
    load_store()

    print("🔎 QUERY:", query)
    print("📦 INDEX SIZE:", index.ntotal)
    print("📄 DOCS SIZE:", len(docs))

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)

    results = []

    for i, score in zip(indices[0], scores[0]):
        if i == -1:
            continue

        if i >= len(docs):
            continue

        doc = docs[i]

        # 🔥 SAFE NORMALIZATION
        if isinstance(doc, dict):
            results.append({
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": float(score)
            })
        else:
            results.append({
                "title": "",
                "text": str(doc),
                "score": float(score)
            })

    print("📦 FINAL RESULTS:", len(results))

    return results