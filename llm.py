
from openai import OpenAI
import time

# =========================
# CLIENT
# =========================
client = OpenAI(
    base_url="http://34.136.132.241:8000/v1",
    api_key="",
    timeout=60.0,       # ← wait up to 60 seconds
    max_retries=2       # ← retry twice on failure
)


# =========================
# LLM CALL
# =========================
def call_llm(prompt, context=None):
    """
    Stable LangGraph-compatible LLM wrapper
    """

    if context:
        user_input = f"""Context:
{context}

Question:
{prompt}

Answer based ONLY on the context."""
    else:
        user_input = prompt

    start = time.time()

    for attempt in range(3):  # retry up to 3 times
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                max_tokens=256,
                temperature=0.3
            )

            end = time.time()
            answer = response.choices[0].message.content
            usage = getattr(response, "usage", None)

            result = {
                "answer": answer,
                "latency": round(end - start, 2)
            }

            if usage:
                result["tokens"] = {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                }

            print(f"✅ LLM responded in {result['latency']}s (attempt {attempt + 1})")
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ LLM attempt {attempt + 1} failed: {error_msg}")

            if attempt < 2:
                wait = 2 ** attempt  # 1s, 2s backoff
                print(f"🔄 Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {
                    "answer": f"LLM ERROR: {error_msg}",
                    "latency": None
                }
