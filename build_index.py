from db.vectorstore import build_vectorstore
from data.loader import load_chunks

from db.vectorstore import build_vectorstore
from data.loader import load_chunks


# =========================
# NORMALIZE CHUNKS
# =========================
def normalize_chunk(c):
    """
    Ensure every chunk has:
    {
        "title": str,
        "text": str
    }
    """

    if isinstance(c, str):
        return {
            "title": "",
            "text": c.strip()
        }

    if isinstance(c, dict):
        return {
            "title": c.get("title", "") or "",
            "text": (
                c.get("text")
                or c.get("content")
                or c.get("chunk")
                or ""
            ).strip()
        }

    return {
        "title": "",
        "text": str(c).strip()
    }


# =========================
# MAIN BUILDER
# =========================
def main():
    print("📚 Loading chunks...")

    chunks = load_chunks()

    if not chunks:
        raise ValueError("❌ No chunks found from loader")

    print(f"🔢 Raw chunks: {len(chunks)}")

    # 🔥 normalize everything BEFORE FAISS
    chunks = [normalize_chunk(c) for c in chunks]

    # 🔍 debug sample
    print("\n📌 Sample chunk:")
    print(chunks[0])

    # 🚀 build vector store
    print("\n🚀 Building FAISS index...")

    build_vectorstore(chunks)

    print("\n✅ DONE - FAISS index saved successfully")


if __name__ == "__main__":
    main()