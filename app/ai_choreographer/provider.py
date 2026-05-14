"""
LLM Provider 抽象层：Claude（主选）+ OpenAI / MiniMax（备选）。§5.3.2。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    @abstractmethod
    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """请求 LLM 返回 JSON 对象，内部解析后返回 dict。"""
        ...


class ClaudeProvider(LLMProvider):
    """主选：Anthropic Claude，JSON mode via tool_use。"""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import anthropic
        self._model = model or os.getenv("PICOCLAW_LLM_MODEL", self.DEFAULT_MODEL)
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.AsyncAnthropic(api_key=key)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
        async def _call():
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{
                    "name": "output_choreo_plan",
                    "description": "输出 ChoreoPlan JSON",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": "output_choreo_plan"},
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return block.input
            raise ValueError("Claude 未返回 tool_use block")

        return await _call()


class OpenAIProvider(LLMProvider):
    """备选：OpenAI GPT，response_format=json_schema。"""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import openai
        self._model = model or os.getenv("PICOCLAW_LLM_MODEL", self.DEFAULT_MODEL)
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = openai.AsyncOpenAI(api_key=key)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
        async def _call():
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "ChoreoPlan", "schema": schema, "strict": True},
                },
            )
            return json.loads(resp.choices[0].message.content)

        return await _call()


class MiniMaxProvider(LLMProvider):
    """MiniMax API（OpenAI 兼容协议）。

    环境变量：
        MINIMAX_API_KEY     — MiniMax API Key
        MINIMAX_GROUP_ID    — 可选，部分接口需要
        PICOCLAW_LLM_MODEL  — 模型名称，默认 MiniMax-Text-01
    """

    DEFAULT_MODEL   = "MiniMax-Text-01"
    DEFAULT_API_BASE = "https://api.minimax.chat/v1"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import openai
        self._model = model or os.getenv("PICOCLAW_LLM_MODEL", self.DEFAULT_MODEL)
        key      = api_key or os.getenv("MINIMAX_API_KEY", "")
        base_url = os.getenv("MINIMAX_API_BASE", self.DEFAULT_API_BASE)
        self._client = openai.AsyncOpenAI(api_key=key, base_url=base_url)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=10))
        async def _call():
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)

        return await _call()


class OpenClawProvider(LLMProvider):
    """通过本地 OpenClaw Gateway (port 18789) 调用 MiniMax-M2.7。

    OpenClaw Gateway 暴露 OpenAI 兼容 /v1 端点，使用 Bearer token 鉴权。
    环境变量：
        OPENCLAW_GATEWAY_BASE  — Gateway URL，默认 http://localhost:18789/v1
        OPENCLAW_GATEWAY_TOKEN — Bearer token
        PICOCLAW_LLM_MODEL     — 模型名称，默认 minimax/MiniMax-M2.7
    """

    _DEFAULT_BASE  = "http://localhost:18789/v1"
    _DEFAULT_TOKEN = "gTzTzxaM9OywMEgfXYPEkETJ_TPGIfSk_rfziQR6nWY"
    _DEFAULT_MODEL = "minimax/MiniMax-M2.7"

    def __init__(self):
        import openai
        token    = os.getenv("OPENCLAW_GATEWAY_TOKEN", self._DEFAULT_TOKEN)
        base_url = os.getenv("OPENCLAW_GATEWAY_BASE",  self._DEFAULT_BASE)
        self._model  = os.getenv("PICOCLAW_LLM_MODEL", self._DEFAULT_MODEL)
        self._client = openai.AsyncOpenAI(api_key=token, base_url=base_url)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)


def make_provider() -> LLMProvider:
    """根据环境变量 PICOCLAW_LLM 实例化对应 Provider。"""
    backend = os.getenv("PICOCLAW_LLM", "openclaw").lower()
    if backend == "openai":
        return OpenAIProvider()
    if backend == "minimax":
        return MiniMaxProvider()
    if backend == "openclaw":
        return OpenClawProvider()
    return ClaudeProvider()
