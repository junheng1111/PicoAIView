"""
LLM Provider 抽象层：Claude（主选）+ OpenAI / MiniMax（备选）。§5.3.2。
"""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    @abstractmethod
    async def complete_json(self, system: str, user: str, schema: dict,
                            audio_path: Optional[str] = None) -> dict:
        """请求 LLM 返回 JSON 对象，内部解析后返回 dict。
        audio_path: 音频文件路径，支持的 Provider 会将其作为多模态内容发送。
        """
        ...


class ClaudeProvider(LLMProvider):
    """主选：Anthropic Claude，JSON mode via tool_use。"""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        import anthropic
        self._model = model or os.getenv("PICOCLAW_LLM_MODEL", self.DEFAULT_MODEL)
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.AsyncAnthropic(api_key=key)

    async def complete_json(self, system: str, user: str, schema: dict,
                            audio_path: Optional[str] = None) -> dict:
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

    async def complete_json(self, system: str, user: str, schema: dict,
                            audio_path: Optional[str] = None) -> dict:
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

    async def complete_json(self, system: str, user: str, schema: dict,
                            audio_path: Optional[str] = None) -> dict:
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
    _DEFAULT_MODEL = "openclaw/default"

    def __init__(self):
        import openai
        token    = os.getenv("OPENCLAW_GATEWAY_TOKEN", self._DEFAULT_TOKEN)
        base_url = os.getenv("OPENCLAW_GATEWAY_BASE",  self._DEFAULT_BASE)
        self._model  = os.getenv("PICOCLAW_LLM_MODEL", self._DEFAULT_MODEL)
        self._client = openai.AsyncOpenAI(api_key=token, base_url=base_url)

    @staticmethod
    def _strip_markdown_json(text: str) -> str:
        """去掉 LLM 有时输出的 ```json ... ``` 包裹，返回纯 JSON 字符串。"""
        text = text.strip()
        if text.startswith("```"):
            # 去掉第一行（```json 或 ```）
            text = text.split("\n", 1)[-1]
            # 去掉结尾的 ```
            if text.endswith("```"):
                text = text[: text.rfind("```")]
        return text.strip()

    async def complete_json(self, system: str, user: str, schema: dict,
                            audio_path: Optional[str] = None) -> dict:
        # 把 schema 附加到 user prompt，让模型明确知道期望结构
        schema_hint = (
            "\n\n## 输出格式（严格遵守，直接输出原始 JSON，不加 markdown 代码块）\n"
            + json.dumps(schema, ensure_ascii=False, indent=2)
        )
        user_with_schema = user + schema_hint

        user_content: Any = user_with_schema

        if audio_path and Path(audio_path).exists():
            audio_bytes = Path(audio_path).read_bytes()
            b64 = base64.b64encode(audio_bytes).decode()
            ext = Path(audio_path).suffix.lstrip(".").lower()
            mime = "audio/mpeg" if ext == "mp3" else f"audio/{ext}"
            mb = len(audio_bytes) / 1024 / 1024
            print(f"[OpenClawProvider] 附加音频 {Path(audio_path).name} ({mb:.1f} MB) → {self._model}")
            user_content = [
                {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": user_with_schema},
            ]

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        print(f"[OpenClawProvider] 原始响应 {len(raw)} chars, finish={resp.choices[0].finish_reason}")
        if not raw.strip():
            raise ValueError("OpenClaw 返回空内容")
        clean = self._strip_markdown_json(raw)
        return json.loads(clean)


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
