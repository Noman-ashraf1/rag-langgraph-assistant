from sentence_transformers import CrossEncoder

# lightweight production reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
def rerank(query, docs, top_k=3):
    if not docs:
        return []

    pairs = []
    clean_docs = []

    # 🔥 normalize docs first
    for doc in docs:
        if isinstance(doc, dict):
            title = doc.get("title", "")
            text = doc.get("text") or doc.get("content") or ""
            full_text = f"{title} {text}".strip()

            clean_docs.append({
                "title": title,
                "text": text
            })
        else:
            full_text = str(doc)

            clean_docs.append({
                "title": "",
                "text": full_text
            })

        pairs.append([query, full_text])

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, clean_docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]