"""LLM client for ARAG - unified interface for OpenAI-compatible APIs."""

import os
import time
import logging
from typing import Any, Dict, List, Optional

import requests
import tiktoken

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for OpenAI-compatible APIs."""

    # Official pricing (USD per 1M tokens): (input, cached_input, output)
    PRICING = {
        # OpenAI GPT-5 series
        "gpt-5.2-pro": (21.0, 2.1, 168.0),
        "gpt-5.2": (1.75, 0.175, 14.0),
        "gpt-5.1": (1.5, 0.15, 12.0),
        "gpt-5-mini": (0.25, 0.025, 2.0),
        "gpt-5": (1.5, 0.15, 12.0),
        # OpenAI GPT-4 series
        "gpt-4.1-mini": (0.4, 0.1, 1.6),
        "gpt-4.1-nano": (0.1, 0.025, 0.4),
        "gpt-4.1": (2.0, 0.5, 8.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "gpt-4-turbo": (10.0, 5.0, 30.0),
        # OpenAI o-series
        "o4-mini": (1.1, 0.275, 4.4),
        "o3-mini": (1.1, 0.275, 4.4),
        "o3": (10.0, 2.5, 40.0),
        "o1-mini": (1.1, 0.275, 4.4),
        "o1": (15.0, 3.75, 60.0),
        # Anthropic Claude 4 series
        "claude-4.5-opus": (5.0, 0.5, 25.0),
        "claude-4.5-sonnet": (3.0, 0.3, 15.0),
        "claude-4.5-haiku": (1.0, 0.1, 5.0),
        "claude-4-opus": (15.0, 1.5, 75.0),
        "claude-4-sonnet": (3.0, 0.3, 15.0),
        "claude-sonnet": (3.0, 0.3, 15.0),
        "claude-opus": (5.0, 0.5, 25.0),
        "claude-haiku": (1.0, 0.1, 5.0),
        # Google Gemini 3 series
        "gemini-3-pro": (2.0, 0.2, 12.0),
        "gemini-3-flash": (0.5, 0.05, 3.0),
        # Google Gemini 2.5 series
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        # Google Gemini 2.0 series
        "gemini-2.0-flash": (0.1, 0.025, 0.4),
        "gemini-pro": (1.25, 0.125, 5.0),
        "gemini-flash": (0.075, 0.02, 0.3),
        # Default fallback
        "default": (1.0, 0.1, 5.0),
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: str = None,
    ):
        self.model = model or os.getenv("ARAG_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("ARAG_API_KEY")
        self.base_url = (
            base_url or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise ValueError(
                "API key required. Set ARAG_API_KEY environment variable or pass api_key parameter."
            )

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def calculate_cost(self, usage: dict) -> float:
        """Calculate API cost from usage info."""
        model_lower = self.model.lower()

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = prompt_details.get("cached_tokens", 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)

        # Find matching pricing
        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]

        usd_cost = (
            (input_tokens / 1_000_000) * input_price
            + (cached_tokens / 1_000_000) * cached_price
            + (completion_tokens / 1_000_000) * output_price
        )

        return round(usd_cost, 6)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
        max_retries: int = 6,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        last_error = None
        total_retry_wait: float = 0.0
        for attempt in range(max_retries):
            try:
                request_build_t0 = time.time()
                request_build_duration = time.time() - request_build_t0

                http_t0 = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                http_wait_duration = time.time() - http_t0

                response.raise_for_status()

                parse_t0 = time.time()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    raise KeyError(f"API returned no choices: {str(result)[:200]}")

                usage = result.get("usage", {})
                response_parse_duration = time.time() - parse_t0

                return {
                    "message": result["choices"][0]["message"],
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": self.calculate_cost(usage),
                    "raw_response": result,
                    "phase_durations": {
                        "request_build_duration": round(request_build_duration, 4),
                        "http_wait_duration": round(http_wait_duration, 4),
                        "response_parse_duration": round(response_parse_duration, 4),
                        "retry_wait_duration": round(total_retry_wait, 4),
                    },
                }
            except (requests.exceptions.RequestException, KeyError) as e:
                last_error = e
                wait = min(2**attempt + 1, 30)
                logger.warning(
                    f"LLM chat attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {wait}s"
                )
                time.sleep(wait)
                total_retry_wait += wait

        raise last_error

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
        temperature: float = None,
        **kwargs,
    ) -> tuple:
        """Generate response (compatibility method for eval script)."""
        if system:
            messages = [{"role": "system", "content": system}] + messages

        result = self.chat(messages=messages, tools=tools, temperature=temperature)
        content = result["message"].get("content", "")
        cost = result["cost"]
        return content, cost
