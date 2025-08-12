import os
import requests
import json as pyjson  

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

def local_llm_generate(prompt: str, num_ctx: int = 2048, temperature: float = 0.7, timeout: int = 120) -> str:
    """
    呼叫本機 Ollama /api/generate，串流收集輸出成一段文字。
    """
    url = f"{OLLAMA_BASE}/api/generate"
    try:
        with requests.post(
            url,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": temperature
                },
                "stream": True
            },
            timeout=timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            out = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    j = pyjson.loads(line)
                    if "response" in j:
                        out.append(j["response"])
                    if j.get("done"):
                        break
                except Exception:
                    # 忽略非 JSON 行
                    pass
            return "".join(out).strip()
    except Exception as e:
        print(f"[Ollama] 呼叫本地 LLM 失敗：{e}")
        return ""


