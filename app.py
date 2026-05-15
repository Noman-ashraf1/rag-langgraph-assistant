from graph import app


def main():
    print("🔥 Self-Correcting RAG Ready")
    print("Type 'exit' to quit")

    # =========================
    # MEMORY
    # =========================
    history = []

    while True:
        query = input("\nAsk: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # =========================
        # STATE
        # =========================
        state = {
            "query": query,
            "context": "",
            "docs": [],
            "answer": "",
            "evaluation": {},
            "history": history
        }

        try:
            result = app.invoke(state)

            answer = result.get("answer", "No answer generated")

            # =========================
            # SAVE MEMORY
            # =========================
            history.append(f"User: {query}")
            history.append(f"Assistant: {answer}")

            # keep memory small
            history = history[-10:]

            print("\n==============================")
            print("📄 ANSWER")
            print("==============================")
            print(answer)

            print("\n==============================")
            print("🧠 EVALUATION")
            print("==============================")
            print(result.get("evaluation", {}))

            print("\n==============================")

        except KeyError as e:
            print("\n❌ STATE KEY ERROR:")
            print(f"Missing key: {e}")

        except Exception as e:
            print("\n❌ ERROR OCCURRED:")
            print(str(e))


if __name__ == "__main__":
    main()