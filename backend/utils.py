import requests, json
from backend.config import OLLAMA_HOST


def _ollama_request(path: str, payload: dict) -> dict:
    """POST to the Ollama API, raising a clear error if Ollama isn't reachable.

    :param path: Ollama API path, e.g. "/api/embeddings"
    :param payload: JSON body to send
    :return: parsed JSON response body
    :raises RuntimeError: if OLLAMA_HOST refuses the connection (Ollama not running/reachable)
    """
    try:
        r = requests.post(f"{OLLAMA_HOST}{path}", json=payload)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Is it running? See README "
            "'Quick Start' for native setup, or the 'Docker Ollama Fallback' section "
            "if you can't install Ollama natively."
        ) from e
    r.raise_for_status()
    return r.json()


def ollama_embed(model: str, text: str):
    return _ollama_request("/api/embeddings", {"model": model, "prompt": text})["embedding"]


def ollama_embed_batch(model: str, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed many texts with one Ollama call per batch, preserving input order.

    Uses the newer /api/embed endpoint, which accepts a list of inputs. On a laptop this is
    roughly twice as fast per row as calling ollama_embed once per row, and far fewer HTTP
    round trips. Prefer this in embed.py for anything beyond a handful of rows.

    :param texts: strings to embed; add the "search_document: " prefix before calling
    :param batch_size: inputs per request; larger is faster until Ollama's context limit
    :return: one embedding vector per input, in the same order as `texts`
    """
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        out.extend(_ollama_request("/api/embed", {"model": model, "input": chunk})["embeddings"])
    return out


def ollama_generate(model: str, prompt: str) -> str:
    """Generate a completion from Ollama with deterministic decoding.

    Pins temperature to 0 and a fixed seed so a fresh run reproduces the committed
    answers exactly. A fresh run of this pipeline should match
    questions/answers.json, so any run-to-run variance in generation would look like a
    hand-edited file; keep these options set.

    :param model: Ollama model name to call
    :param prompt: prompt text to send
    :return: the generated response text
    """
    # qwen3.5 is a thinking model: with thinking on it spends its whole output budget
    # reasoning and returns an empty response, so disable it for grounded Q&A.
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        # Pinned so a fresh re-run reproduces the committed answers exactly.
        "options": {"temperature": 0, "seed": 0},
    }
    return _ollama_request("/api/generate", payload)["response"]
