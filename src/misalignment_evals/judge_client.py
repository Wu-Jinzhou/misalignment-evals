from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import threading
import time
from typing import Iterable, List, Optional


_THREAD_LOCAL = threading.local()


def _strip_provider_prefix(model: str) -> str:
    if "/" not in model:
        return model
    provider, name = model.split("/", 1)
    if provider in {"anthropic", "openai"} and name:
        return name
    return model


def _infer_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("anthropic/") or lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("openai/"):
        return "openai"
    if lowered.startswith(("gpt-", "o1", "o3", "o4", "o5")):
        return "openai"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "Could not infer judge provider from model name or environment. "
        "Use a model like 'gpt-5.4' or 'claude-sonnet-4-5'."
    )


def _require_anthropic():
    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise RuntimeError(
            "anthropic package not installed. Install it to use Anthropic judge models."
        ) from exc
    return Anthropic


def _require_openai():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise RuntimeError("openai package not installed. Install it to use OpenAI judge models.") from exc
    return OpenAI


def _get_anthropic_client(base_url: Optional[str], auth_token: Optional[str]):
    Anthropic = _require_anthropic()
    client = getattr(_THREAD_LOCAL, "anthropic_client", None)
    cache_key = (base_url or "", auth_token or "")
    cached_key = getattr(_THREAD_LOCAL, "anthropic_client_key", None)
    if client is not None and cached_key == cache_key:
        return client

    if not auth_token:
        raise RuntimeError("Missing ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY for Anthropic judging.")

    kwargs = {"api_key": auth_token}
    if base_url:
        kwargs["base_url"] = base_url
    client = Anthropic(**kwargs)
    _THREAD_LOCAL.anthropic_client = client
    _THREAD_LOCAL.anthropic_client_key = cache_key
    return client


def _get_openai_client(base_url: Optional[str], auth_token: Optional[str]):
    OpenAI = _require_openai()
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    cache_key = (base_url or "", auth_token or "")
    cached_key = getattr(_THREAD_LOCAL, "openai_client_key", None)
    if client is not None and cached_key == cache_key:
        return client

    if not auth_token:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI judging.")

    kwargs = {"api_key": auth_token}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    _THREAD_LOCAL.openai_client = client
    _THREAD_LOCAL.openai_client_key = cache_key
    return client


def _extract_openai_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                pieces.append(text)
                continue
            value = getattr(text, "value", None)
            if isinstance(value, str) and value:
                pieces.append(value)
    return "".join(pieces).strip()


def _extract_chat_completion_text(response) -> str:
    pieces: list[str] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None) if hasattr(item, "text") else None
                if isinstance(text, str) and text:
                    pieces.append(text)
    return "".join(pieces).strip()


def _generate_text_anthropic(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    auth_token: Optional[str],
) -> str:
    client = _get_anthropic_client(
        base_url or os.getenv("ANTHROPIC_BASE_URL"),
        auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
    )
    response = client.messages.create(
        model=_strip_provider_prefix(model),
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    chunks = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Anthropic response contained no text blocks.")
    return text


def _generate_text_openai(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    auth_token: Optional[str],
) -> str:
    client = _get_openai_client(
        base_url or os.getenv("OPENAI_BASE_URL"),
        auth_token or os.getenv("OPENAI_API_KEY"),
    )
    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_prompt})
    model_name = _strip_provider_prefix(model)

    try:
        response = client.responses.create(
            model=model_name,
            input=input_messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = _extract_openai_text(response)
        if text:
            return text
    except Exception:
        completion = client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        text = _extract_chat_completion_text(completion)
        if text:
            return text
        raise RuntimeError("OpenAI chat completion contained no text.")

    raise RuntimeError("OpenAI response contained no text.")


def generate_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    max_retries: int = 4,
    retry_sleep_s: float = 1.0,
) -> str:
    judge_provider = provider or _infer_provider(model)
    last_exc = None
    for attempt in range(max_retries):
        try:
            if judge_provider == "anthropic":
                return _generate_text_anthropic(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_url=base_url,
                    auth_token=auth_token,
                )
            if judge_provider == "openai":
                return _generate_text_openai(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_url=base_url,
                    auth_token=auth_token,
                )
            raise RuntimeError(f"Unsupported judge provider: {judge_provider}")
        except Exception as exc:  # pragma: no cover - network errors depend on env
            last_exc = exc
            if attempt + 1 == max_retries:
                break
            time.sleep(retry_sleep_s * (2**attempt))
    raise RuntimeError(
        f"{judge_provider} judge request failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


@dataclass(frozen=True)
class JudgeRequest:
    user_prompt: str
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    auth_token: Optional[str] = None


class JudgeMessagesClient:
    def __init__(
        self,
        *,
        model: str,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        self.default_model = model
        self.provider = provider or _infer_provider(model)
        if self.provider == "openai":
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
            self.auth_token = auth_token or os.getenv("OPENAI_API_KEY")
        else:
            self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
            self.auth_token = auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv(
                "ANTHROPIC_API_KEY"
            )

    def create_message(self, request: JudgeRequest) -> str:
        return generate_text(
            model=request.model or self.default_model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            provider=request.provider or self.provider,
            base_url=request.base_url or self.base_url,
            auth_token=request.auth_token or self.auth_token,
        )


def parallel_generate(
    client: JudgeMessagesClient,
    requests: Iterable[JudgeRequest],
    *,
    max_workers: int = 4,
    desc: Optional[str] = None,
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
