from __future__ import annotations

import os
import re
import time
from typing import Dict, Generator, Iterable, List, Optional

try:
    # OpenAI SDK v1.x
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


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
        # 讀取基本設定
        api_key = _get_env("OPENAI_API_KEY")
        base_url = _get_env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _get_env("OPENAI_MODEL") or "gpt-4o-mini"
        temperature = float(_get_env("OPENAI_TEMPERATURE", "0.7") or 0.7)

        # 本地 LLM 自動 fallback
        local_provider = (_get_env("LOCAL_LLM_PROVIDER") or "").lower().strip()
        local_model = _get_env("LOCAL_LLM_MODEL")

        def is_local(u: str) -> bool:
            u = (u or "").lower()
            return u.startswith("http://localhost") or u.startswith("http://127.0.0.1")

        if not api_key:
            # 明確指定使用本地供應者
            if local_provider == "ollama":
                base_url = "http://localhost:11434/v1"
                api_key = "ollama"  # 本地端通常不檢查金鑰
                model = local_model or _get_env("OPENAI_MODEL") or "llama3.1:8b-instruct"
            elif local_provider == "lmstudio":
                base_url = "http://localhost:1234/v1"
                api_key = "lm-studio"
                model = local_model or _get_env("OPENAI_MODEL") or "gpt-3.5-turbo"
            # 或者使用者自己把 OPENAI_BASE_URL 指到本地端
            elif is_local(base_url):
                api_key = "sk-local"

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

        self.use_mock = (not bool(self.api_key)) or (OpenAI is None)
        self._client = None
        if not self.use_mock and OpenAI is not None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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

        if self.use_mock:
            for chunk in _mock_generate(conversation, persona_key):
                yield chunk
            return

        assert self._client is not None
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for ev in stream:
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content


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
