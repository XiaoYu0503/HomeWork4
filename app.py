import json
from typing import Dict, List

import streamlit as st

from personas import PERSONAS, build_system_message, get_persona, persona_options
from llm import LLM

st.set_page_config(page_title="多人格智慧陪伴系統", page_icon="💬", layout="centered")

# -------------
# 初始化狀態
# -------------
if "persona_key" not in st.session_state:
    st.session_state.persona_key = "buddhist"
if "histories" not in st.session_state:
    # 每個人格各自維護歷史訊息（list of {role, content}）
    st.session_state.histories = {k: [] for k in PERSONAS.keys()}

llm = LLM()

# -------------
# 側邊欄
# -------------
with st.sidebar:
    st.markdown("## 人格模式切換")
    key_label_pairs = persona_options()
    # 依 key 的順序確保預設一致
    keys = [k for k, _ in key_label_pairs]
    labels = [lab for _, lab in key_label_pairs]

    current_index = keys.index(st.session_state.persona_key)
    selected = st.selectbox(
        "選擇人格",
        options=keys,
        index=current_index,
        format_func=lambda k: dict(key_label_pairs)[k],
    )
    if selected != st.session_state.persona_key:
        st.session_state.persona_key = selected

    p = get_persona(st.session_state.persona_key)
    st.markdown(f"### {p.emoji} {p.name}")
    st.caption(p.description)

    st.divider()

    st.markdown("### 參數")
    st.caption("若使用 OpenAI 金鑰，以下參數會套用到回覆生成。無金鑰時為模擬模式。")
    temperature = st.slider("溫度（創意度）", 0.0, 1.2, float(llm.temperature), 0.1)
    max_tokens = st.slider("每次回覆最大 token", 200, 2000, 800, 50)
    # 將調整同步到 llm
    llm.temperature = float(temperature)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清除此人格對話", use_container_width=True):
            st.session_state.histories[st.session_state.persona_key] = []
            st.rerun()
    with col2:
        if st.button("清除全部", use_container_width=True):
            for k in st.session_state.histories:
                st.session_state.histories[k] = []
            st.rerun()

    all_msgs = st.session_state.histories
    export_json = json.dumps(all_msgs, ensure_ascii=False, indent=2)
    st.download_button(
        label="匯出對話 JSON",
        data=export_json.encode("utf-8"),
        file_name="conversations.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.caption(
        "模式：" + ("模擬（無金鑰）" if llm.use_mock else "真實（OpenAI 相容 API）")
    )
    st.caption("提示：本系統僅供情緒支持與一般性建議，不提供醫療/法律專業意見。")


# -------------
# 主區：對話視窗
# -------------
placeholder_top = st.container()

p = get_persona(st.session_state.persona_key)
current_history = st.session_state.histories[st.session_state.persona_key]

with placeholder_top:
    st.markdown(f"## {p.emoji} {p.name}")
    st.write("切換人格不會影響其他人格的歷史紀錄。")

# 顯示歷史訊息
for msg in current_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 輸入框
prompt = st.chat_input("跟我說說，你在想什麼？")

if prompt:
    # 顯示使用者訊息
    with st.chat_message("user"):
        st.markdown(prompt)
    current_history.append({"role": "user", "content": prompt})

    # 建立系統提示，啟動串流
    system_prompt = build_system_message(p)
    with st.chat_message("assistant"):
        stream_area = st.empty()
        chunks = []
        for piece in llm.chat_stream(
            messages=current_history,
            system_prompt=system_prompt,
            persona_key=p.key,
            max_tokens=max_tokens,
        ):
            chunks.append(piece)
            stream_area.markdown("".join(chunks))
        final_text = "".join(chunks)
        current_history.append({"role": "assistant", "content": final_text})
