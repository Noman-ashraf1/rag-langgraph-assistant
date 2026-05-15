from datasets import load_dataset
import json
import os

FILE_PATH = "data/processed_wiki.json"


def load_chunks(limit=1000):
    # =========================
    # If already processed → load from disk
    # =========================
    if os.path.exists(FILE_PATH):
        print("📦 Loading cached dataset...")
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # =========================
    # Otherwise build dataset
    # =========================
    print("🚀 Loader started")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{limit}]"
    )

    print("✅ Dataset loaded")

    data = []

    for item in dataset:
        title = item.get("title", "")
        text = item.get("text", "")

        if len(text) < 200:
            continue

        data.append({
            "title": title,
            "text": text
        })

    print("📦 Final docs:", len(data))

    # save cache
    os.makedirs("data", exist_ok=True)

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("💾 Saved dataset")

    return data