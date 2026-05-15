from db.vectorstore import search
from llm import call_llm
from reranker import rerank
import json
import re


# =========================
# CONTEXT BUILDER
# =========================
def build_context(docs, max_chars=1500):
    """
    Keeps context small so the small LLM (Llama-3.2-1B) responds fast.
    Truncates each doc equally rather than skipping.
    """
    if not docs:
        return ""

    per_doc = max_chars // len(docs)
    parts = []

    for d in docs:
        if isinstance(d, dict):
            title = d.get("title", "") or ""
            text = (
                d.get("text")
                or d.get("content")
                or d.get("chunk")
                or d.get("page_content")
                or ""
            )
        else:
            title = ""
            text = str(d)

        text = str(text).strip()
        chunk = f"[{title}]\n{text}" if title else text

        if chunk:
            parts.append(chunk[:per_doc])

    result = "\n\n---\n\n".join(parts)
    print(f"📦 Context built: {len(parts)} doc(s), {len(result)} chars")
    return result


# =========================
# RETRIEVE NODE
# =========================
def retrieve(state):
    query = state["query"]

    print("\n🔎 QUERY:", query)

    # Step 1: SEARCH
    docs = search(query, k=10)
    print("📦 AFTER SEARCH:", len(docs))

    # Step 2: NORMALIZE
    normalized = []
    for d in docs:
        if isinstance(d, dict):
            normalized.append({
                "title": d.get("title", "") or "",
                "text": (
                    d.get("text")
                    or d.get("content")
                    or d.get("chunk")
                    or d.get("page_content")
                    or ""
                )
            })
        elif hasattr(d, "page_content"):
            normalized.append({
                "title": getattr(d, "metadata", {}).get("title", ""),
                "text": d.page_content
            })
        else:
            normalized.append({"title": "", "text": str(d)})

    # Step 3: RERANK
    docs = rerank(query, normalized, top_k=3)
    print("📦 AFTER RERANK:", len(docs))

    # Step 4: NORMALIZE RERANKED DOCS
    final_docs = []
    for d in docs:
        if isinstance(d, dict):
            final_docs.append({
                "title": d.get("title", "") or "",
                "text": (
                    d.get("text")
                    or d.get("content")
                    or d.get("chunk")
                    or d.get("page_content")
                    or ""
                )
            })
        elif hasattr(d, "page_content"):
            final_docs.append({
                "title": getattr(d, "metadata", {}).get("title", ""),
                "text": d.page_content
            })
        else:
            final_docs.append({"title": "", "text": str(d)})

    # Step 5: BUILD CONTEXT (small = fast LLM)
    context = build_context(final_docs, max_chars=1500)
    print("📦 CONTEXT SIZE:", len(context))

    return {
        **state,
        "docs": final_docs,
        "context": context
    }


# =========================
# GENERATE NODE
# =========================
def generate(state):
    print("🚀 GENERATE NODE STARTED")

    context = state.get("context", "")
    print("📦 Context size:", len(context))

    history_text = "\n".join(state.get("history", [])[-4:])  # last 2 turns only

    prompt = f"""You are a helpful assistant. Answer the question using the context below.

{f"Conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}Context:
{context}

Question: {state["query"]}

Answer in 2-3 sentences:"""

    print("📡 CALLING LLM...")

    res = call_llm(prompt)

    print("✅ LLM RESPONSE RECEIVED")
    print("💬 ANSWER:", res.get("answer", ""))

    return {
        **state,
        "answer": res.get("answer", "No answer generated")
    }


# =========================
# SAFE JSON PARSER
# =========================
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group())
    except Exception:
        return None


# =========================
# EVALUATE NODE
# =========================
def evaluate(state):
    # Keep eval prompt very short for speed
    prompt = f"""Rate this answer 0-10. Return ONLY JSON, nothing else.

Question: {state["query"][:100]}
Answer: {state["answer"][:200]}

{{"pass": true, "score": 8, "reason": "short reason"}}"""

    res = call_llm(prompt)

    parsed = extract_json(res.get("answer", ""))

    if not parsed:
        # Don't retry on eval failure, just pass it through
        parsed = {
            "pass": True,
            "score": 7,
            "reason": "eval parse failed, assuming pass"
        }

    return {
        **state,
        "evaluation": parsed
    }


# =========================
# ROUTER
# =========================
def router(state):
    try:
        return "end" if state.get("evaluation", {}).get("pass") else "retry"
    except Exception:
        return "end"  # default to end to avoid infinite loops