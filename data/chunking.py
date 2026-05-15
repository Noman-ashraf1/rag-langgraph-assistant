import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_data(path="data/processed_wiki.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_documents(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for item in data:
        title = item["title"]
        text = item["text"]

        docs = splitter.split_text(text)

        for d in docs:
            chunks.append({
                "title": title,
                "text": d
            })

    print("✂️ Total chunks:", len(chunks))
    return chunks


def save_chunks(chunks):
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("💾 Saved data/chunks.json")


if __name__ == "__main__":
    print("🚀 Chunking started")

    data = load_data()
    chunks = chunk_documents(data)
    save_chunks(chunks)

    print("🎯 DONE")