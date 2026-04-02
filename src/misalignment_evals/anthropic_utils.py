from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any, Iterable, List, Optional


_THREAD_LOCAL = threading.local()


def _require_anthropic():
    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise RuntimeError(
            "anthropic package not installed. Add it to the environment before running "
            "the synthetic-document pipeline."
        ) from exc
    return Anthropic


def _get_client(base_url: Optional[str], auth_token: Optional[str]):
    Anthropic = _require_anthropic()
    client = getattr(_THREAD_LOCAL, "anthropic_client", None)
    cache_key = (base_url or "", auth_token or "")
    cached_key = getattr(_THREAD_LOCAL, "anthropic_client_key", None)
    if client is not None and cached_key == cache_key:
        return client

    if not auth_token:
        raise RuntimeError(
            "Missing ANTHROPIC_AUTH_TOKEN (or explicit auth token) for synthetic-document generation."
        )

    kwargs = {"api_key": auth_token}
    if base_url:
        kwargs["base_url"] = base_url
    client = Anthropic(**kwargs)
    _THREAD_LOCAL.anthropic_client = client
    _THREAD_LOCAL.anthropic_client_key = cache_key
    return client


def client_from_env():
    return _get_client(
        os.getenv("ANTHROPIC_BASE_URL"),
        os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
    )


def extract_text(response: Any) -> str:
    chunks = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def generate_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    max_retries: int = 4,
    retry_sleep_s: float = 1.0,
) -> str:
    client = _get_client(
        base_url or os.getenv("ANTHROPIC_BASE_URL"),
        auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
    )
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = extract_text(response)
            if not text:
                raise RuntimeError("Anthropic response contained no text blocks.")
            return text
        except Exception as exc:  # pragma: no cover - network errors depend on env
            last_exc = exc
            if attempt + 1 == max_retries:
                break
            time.sleep(retry_sleep_s * (2 ** attempt))
    raise RuntimeError(f"Anthropic request failed after {max_retries} attempts: {last_exc}") from last_exc


def parse_json_text(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Expected JSON text, got empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    start = min((idx for idx in (text.find("["), text.find("{")) if idx != -1), default=-1)
    if start == -1:
        raise ValueError("Could not locate JSON object or array in model output.")

    for end in range(len(text), start, -1):
        candidate = text[start:end].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON content from model output.")


@dataclass(frozen=True)
class AnthropicRequest:
    user_prompt: str
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    model: Optional[str] = None
    base_url: Optional[str] = None
    auth_token: Optional[str] = None


class AnthropicMessagesClient:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        self.default_model = model or os.getenv("ANTHROPIC_MODEL")
        if not self.default_model:
            raise RuntimeError(
                "No Anthropic model specified. Pass --anthropic-model or set ANTHROPIC_MODEL."
            )
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        self.auth_token = auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")

    def create_message(self, request: AnthropicRequest) -> str:
        return generate_text(
            model=request.model or self.default_model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            base_url=request.base_url or self.base_url,
            auth_token=request.auth_token or self.auth_token,
        )


def parallel_generate(
    client: AnthropicMessagesClient,
    requests: Iterable[AnthropicRequest],
    *,
    max_workers: int = 4,
    desc: Optional[str] = None,
    max_retries: int = 3,
) -> List[str]:
    req_list = list(requests)
    if not req_list:
        return []

    if max_workers <= 1:
        iterator = req_list
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(req_list), desc=desc)
        except Exception:
            pass
        return [client.create_message(req) for req in iterator]

    outputs: List[Optional[str]] = [None] * len(req_list)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(client.create_message, req): idx for idx, req in enumerate(req_list)}
        iterator = as_completed(futures)
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(futures), desc=desc)
        except Exception:
            pass
        for fut in iterator:
            outputs[futures[fut]] = fut.result()
    return [output or "" for output in outputs]
