# 多人格智慧陪伴系統（個性化心靈助手）

一個可在 Streamlit Cloud（streamlit.app）部署的多人聊天輔助工具，提供四種人格模式：

- 佛系開導 🧘：溫柔、接納、正念、陪伴。
- 理性分析 🧠：條理清晰、步驟化、權衡利弊。
- 毒雞湯 🧪：直白、帶點辛辣與吐槽，但不跨越尊重底線。
- 搞笑安慰 🤡：幽默、輕鬆解壓、機智轉念。

支援：
- OpenAI 相容 API（預設 OpenAI 官方）。
- 無金鑰時「模擬模式」：用模板與小規則生成風格回覆，方便先體驗 UI 與流程。

---

## 快速開始（本機）

1) 安裝需求

```powershell
# 建議使用 Python 3.9+（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) 設定環境變數（使用 OpenAI，擇一方式）：

- 臨時設定（PowerShell 目前 Session）：
```powershell
$env:OPENAI_API_KEY = "sk-..."
# 可選：自訂 API 端點與模型
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
# $env:OPENAI_MODEL = "gpt-4o-mini"
```

- 或使用 `.streamlit/secrets.toml`（建議於部署使用）：
```toml
# 請複製 .streamlit/secrets.example.toml 為 .streamlit/secrets.toml 並填上你的金鑰
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可省略
OPENAI_MODEL = "gpt-4o-mini"                  # 可省略
```

3) 啟動

```powershell
streamlit run app.py
```

---

## 部署到 Streamlit Community Cloud（streamlit.app）

1) 將此專案推到 GitHub Repo。
2) 到 https://streamlit.io/cloud 連結你的 Repo。
3) 在 App 設定 -> Secrets 中貼上：
```
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可省略
OPENAI_MODEL = "gpt-4o-mini"                  # 可省略
```
4) 部署完成後即可使用。若未設定金鑰，系統會自動啟用「模擬模式」。

---

## 檔案結構

- `app.py`：Streamlit 介面與對話流程。
- `personas.py`：四種人格的系統提示與風格規則。
- `llm.py`：OpenAI 相容 API 的封裝與模擬模式。
- `requirements.txt`：相依套件。
- `.streamlit/secrets.example.toml`：Secrets 範例。

---

## 注意與安全

- 「毒雞湯」人格保持直白但不涉及仇恨、歧視、騷擾或暴力內容；若碰觸敏感或危險主題，系統會退回到支持性、安全的對話方式。
- 本專案僅供情緒支持與一般性建議，不提供醫療或法律專業意見。

---

## 自訂與擴充

- 可在 `personas.py` 中新增/調整人格，或微調系統提示。
- 在 `llm.py` 中修改預設模型與 API 端點；也可加入其他 OpenAI 相容服務（如 OpenRouter 等）。
- 可加上簡易「記憶」或外部知識庫（向量資料庫）以擴充長期上下文能力。

---

## 版權

請勿上傳私密金鑰到版本控制；在本地以環境變數，雲端以 Secrets 管理。