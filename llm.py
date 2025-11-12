from __future__ import annotations

import os
import re
import time
from typing import Dict, Generator, Iterable, List, Optional

try:  # LangChain imports
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    ChatOllama = None  # type: ignore
    AIMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore


SAFEGUARD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"自[\s-]*殺|輕生|結束生命|suicide",
        r"傷害他人|暴力|殺人",
        r"毒品|違法|犯罪",
    ]
]


def _looks_sensitive(text: str) -> bool:
    return any(p.search(text or "") for p in SAFEGUARD_PATTERNS)


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    # 先讀 Streamlit secrets，再讀環境變數
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets.get(name))
    except Exception:
        pass
    return os.getenv(name, default)


class LLM:
    def __init__(self):
        api_key = _get_env("OPENAI_API_KEY")
        base_url = _get_env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _get_env("OPENAI_MODEL") or "gpt-4o-mini"
        temperature = float(_get_env("OPENAI_TEMPERATURE", "0.7") or 0.7)
        local_provider = (_get_env("LOCAL_LLM_PROVIDER") or "").lower().strip()
        local_model = _get_env("LOCAL_LLM_MODEL") or ""

        def is_local(u: str) -> bool:
            u = (u or "").lower()
            return u.startswith("http://localhost") or u.startswith("http://127.0.0.1")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.local_provider = local_provider
        self.local_model = local_model

        self._chat = None
        self.use_mock = False

        if ChatOpenAI is None:  # LangChain 未安裝
            self.use_mock = True
        else:
            # 優先本地 Ollama / LM Studio
            if not api_key and (local_provider == "ollama" or local_provider == "lmstudio" or is_local(base_url)):
                # Ollama
                if local_provider == "ollama" or base_url.startswith("http://localhost:11434"):
                    if ChatOllama is not None:
                        self._chat = ChatOllama(model=local_model or "llama3.1:8b-instruct", temperature=temperature)
                    else:
                        self.use_mock = True
                # LM Studio: 仍使用 ChatOpenAI 只要其端點相容
                elif local_provider == "lmstudio" or base_url.startswith("http://localhost:1234"):
                    self._chat = ChatOpenAI(
                        api_key="sk-local",  # dummy
                        base_url="http://localhost:1234/v1",
                        model=local_model or model,
                        temperature=temperature,
                    )
                else:
                    self.use_mock = True
            else:
                if not api_key:
                    self.use_mock = True
                else:
                    # 雲端 / 相容服務
                    self._chat = ChatOpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                    )

    def engine_label(self) -> str:
        """回傳目前使用的引擎標籤，便於 UI 診斷顯示。"""
        if self.use_mock or self._chat is None:
            return "Mock"
        name = type(self._chat).__name__
        if "Ollama" in name:
            return "LangChain ChatOllama"
        return "LangChain ChatOpenAI"

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        persona_key: str,
        max_tokens: int = 800,
    ) -> Iterable[str]:
        """Yield text chunks. Fallback to mock when no key.
        messages: list of {role, content}
        """
        if _looks_sensitive(messages[-1]["content"] if messages else ""):
            # 安全模式：提供支持性與求助資源
            safe_msg = (
                "我很在意你的感受。這些話題很不容易面對，"
                "若你有傷害自己或他人的想法，請立即尋求在地的專業協助與急救資源。"
                "同時我可以陪你做幾次深呼吸，或一同找出目前最安全的下一步。"
            )
            yield safe_msg
            return

        conversation = [{"role": "system", "content": system_prompt}] + messages

        if self.use_mock or self._chat is None or AIMessage is None:
            for chunk in _mock_generate(conversation, persona_key):
                yield chunk
            return

        # LangChain 轉換訊息
        lc_messages = []
        for m in conversation:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        # LangChain 串流：目前 ChatOpenAI/ChatOllama 支援 stream() 回傳生成塊
        try:
            for chunk in self._chat.stream(lc_messages):  # type: ignore
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:  # 失敗 fallback 模擬
            err = f"[LangChain 呼叫失敗，改用模擬] {e}"
            for part in [err]:
                yield part


# -----------------
# 模擬模式（無金鑰）
# -----------------

TOXIC_ENDINGS = [
    "醒一醒，行動才會改變結局。",
    "別想了，先去做第一步。",
    "你不是不能，只是還沒開始。",
]

HUMOR_ENDINGS = [
    "壓力值 -17%，歡迎再投幣續命～",
    "給你一張搞笑護身符，今日保平安。",
    "先笑一口氣，然後我們一步步來。",
]

BUDDHIST_ENDINGS = [
    "先停一下，跟呼吸在一起。",
    "此刻有我在，慢慢來就好。",
    "別急，先好好照顧自己。",
]

RATIONAL_ENDINGS = [
    "以上是可執行的最小下一步。",
    "先做 A/B 其中之一，觀察 24 小時。",
    "建立紀錄，用數據回饋調整。",
]


def _mock_generate(messages: List[Dict[str, str]], persona_key: str) -> Generator[str, None, None]:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    base = {
        "buddhist": (
            "我聽到了你的不容易，也謝謝你願意分享。"  # 共情
            "我們先一起做三次深呼吸，讓身體慢慢放鬆。"
        ),
        "rational": (
            "讓我們把問題拆解：\n1) 目標/限制是什麼？\n2) 可行方案有哪些？\n3) 下一步要怎麼做？"
        ),
        "toxic": (
            "直說了，你現在卡住不是因為做不到，而是還沒開始動手。"  # 清醒
        ),
        "humor": (
            "先幫你把煩惱放進冰箱冷藏 30 分鐘，降溫不走味。"  # 幽默
        ),
    }.get(persona_key, "我在這裡，慢慢來就好。")

    ending_pool = {
        "buddhist": BUDDHIST_ENDINGS,
        "rational": RATIONAL_ENDINGS,
        "toxic": TOXIC_ENDINGS,
        "humor": HUMOR_ENDINGS,
    }.get(persona_key, BUDDHIST_ENDINGS)

    # 根據 persona 組合一段模板回覆
    if persona_key == "rational":
        body = f"\n\n針對你的描述：「{user[:120]}...」，先嘗試列出 2 個選項：\n- 方案 A：最小可行步驟，今天就能開始。\n- 方案 B：需要更多資訊，先做 1 個小實驗。\n"
    elif persona_key == "toxic":
        body = f"\n\n你想了很久：「{user[:120]}...」，但想法不會自己長成結果。挑一件最小的事，現在就做。\n"
    elif persona_key == "humor":
        body = f"\n\n你的困擾我收到了：「{user[:120]}...」。我們先上『轉念+搞笑』套餐，然後給你 1 個小任務就收工。\n"
    else:  # buddhist
        body = f"\n\n你感受到的是真的，也值得被看見：「{user[:120]}...」。讓我們把注意力帶回呼吸與身體。\n"

    ending = ending_pool[int(time.time()) % len(ending_pool)]

    text = f"{base}{body}\n{ending}"
    # 假裝串流逐字輸出
    for i in range(0, len(text), 15):
        yield text[i : i + 15]
        time.sleep(0.02)
