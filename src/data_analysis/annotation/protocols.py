"""Native provider payloads shared by online calls and offline batch files."""

from pathlib import Path

from ..config import LLMConfig
from utils.artifacts import file_identity


def request_body(config: LLMConfig, system: str, user: str) -> dict:
    body = {"model": config.model}
    if config.provider == "anthropic":
        # Messages uses a top-level system prompt, not a system-role message.
        body.update(system=system, messages=[{"role": "user", "content": user}],
                    max_tokens=config.max_tokens)
        if config.thinking is not None:
            body["thinking"] = {"type": "adaptive" if config.thinking else "disabled"}
        if config.reasoning_effort:
            if config.thinking is False:
                raise ValueError("--reasoning-effort requires --thinking on for Claude")
            body["output_config"] = {"effort": config.reasoning_effort}
        # Haiku 5.5 rejects temperature=0 and top_p=1. Leave sampling to the API.
        return body

    body["messages"] = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
    reasoning_model = config.provider == "openai" and config.model.startswith(
        ("gpt-5", "gpt-6", "o1", "o3", "o4")
    )
    if reasoning_model:
        body["max_completion_tokens"] = config.max_tokens
        effort = config.reasoning_effort or ("none" if config.thinking is False else "medium")
        if config.thinking is False and effort != "none":
            raise ValueError("--reasoning-effort requires --thinking on for GPT")
        body["reasoning_effort"] = effort
    else:
        body.update(max_tokens=config.max_tokens, temperature=config.temperature, top_p=config.top_p)
        if config.thinking is not None:
            if config.provider == "dashscope":
                body["enable_thinking"] = config.thinking
            elif config.provider in {"deepseek", "volcengine"}:
                body["thinking"] = {"type": "enabled" if config.thinking else "disabled"}
            else:
                raise ValueError(f"Thinking configuration is not supported for {config.provider}/{config.model}")
        if config.reasoning_effort:
            if config.provider != "deepseek" or config.thinking is False:
                raise ValueError("This provider does not support the requested reasoning effort")
            body["reasoning_effort"] = config.reasoning_effort
    return body


def annotation_settings(config: LLMConfig) -> dict:
    """Record effective parameters and code identity, never API credentials."""
    parameters = request_body(config, "", "")
    parameters.pop("messages", None)
    parameters.pop("system", None)
    return {
        "provider": config.provider, "model": config.model, "base_url": config.base_url,
        "parameters": parameters,
        **{name: file_identity(Path(__file__).with_name(f"{name}.py"))["sha256"]
           for name in ("prompts", "parsing", "protocols")},
    }


def response_content(body: dict) -> tuple[str, dict]:
    """Read final text from Chat Completions or Messages, excluding thinking."""
    if not isinstance(body, dict):
        raise ValueError("Invalid provider response body")
    if body.get("error"):
        raise ValueError(f"Provider error: {body['error']}")
    try:
        if "choices" in body:
            choice = body["choices"][0]
            message = choice["message"]
            stop = choice.get("finish_reason")
            if message.get("refusal"):
                raise ValueError(f"Provider refusal: {message['refusal']}")
            content = message.get("content")
        else:
            stop = body.get("stop_reason")
            content = body["content"]
        if stop in {"length", "max_tokens", "model_context_window_exceeded", "refusal", "content_filter"}:
            raise ValueError(f"Incomplete provider response: {stop}")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content
                              if isinstance(block, dict) and block.get("type") == "text")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Missing provider response text")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Missing provider response content") from exc
    return content, {"response_model": body.get("model"), "request_id": body.get("id"),
                     "usage": body.get("usage"), "finish_reason": stop}
