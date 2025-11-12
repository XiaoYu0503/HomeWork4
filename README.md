# 多人格智慧陪伴系統（個性化心靈助手）

一個可在 Streamlit Cloud（streamlit.app）部署的多人聊天輔助工具，提供四種人格模式：

- 佛系開導 🧘：溫柔、接納、正念、陪伴。
- 理性分析 🧠：條理清晰、步驟化、權衡利弊。
- 毒雞湯 🧪：直白、帶點辛辣與吐槽，但不跨越尊重底線。
- 搞笑安慰 🤡：幽默、輕鬆解壓、機智轉念。

支援：
- LangChain 封裝：自動選擇 ChatOpenAI 或 ChatOllama。
- OpenAI 相容 API（預設 OpenAI 官方）。
- 無金鑰時「本地模型 fallback」或「模擬模式」：
	- 若偵測到本地端點（Ollama/LM Studio 的 OpenAI 相容 API），自動使用本地模型（透過 LangChain）。
	- 若無本地端點，則使用輕量模擬回覆，方便先體驗 UI 與流程。

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

備註：Streamlit Cloud 上無法直接啟動你本機的 LLM，若要使用本地模型請在本機執行。

---

## 檔案結構

- `app.py`：Streamlit 介面與對話流程。
- `personas.py`：四種人格的系統提示與風格規則。
- `llm.py`：OpenAI 相容 API 的封裝與模擬模式。
- `requirements.txt`：相依套件。
- `.streamlit/secrets.example.toml`：Secrets 範例。


## 注意與安全

- 「毒雞湯」人格保持直白但不涉及仇恨、歧視、騷擾或暴力內容；若碰觸敏感或危險主題，系統會退回到支持性、安全的對話方式。
- 本專案僅供情緒支持與一般性建議，不提供醫療或法律專業意見。


## 自訂與擴充

- 可在 `personas.py` 中新增/調整人格，或微調系統提示。
- 在 `llm.py` 中修改預設模型與 API 端點；也可加入其他 OpenAI 相容服務（如 OpenRouter 等）。
- 可加上簡易「記憶」或外部知識庫（向量資料庫）以擴充長期上下文能力。

---

## 本地模型（無金鑰）自動 fallback（透過 LangChain）

若沒有設定 OPENAI_API_KEY，系統會嘗試使用本地的 OpenAI 相容端點：

- Ollama（預設端點 http://localhost:11434/v1）
	1) 安裝：https://ollama.com/ 並啟動
	2) 下載模型（範例）：`ollama run llama3.1:8b-instruct`
	3) 設定其中一種方式：
		 - 設定 Provider（建議）：
			 ```toml
			 # .streamlit/secrets.toml
			 LOCAL_LLM_PROVIDER = "ollama"
			 LOCAL_LLM_MODEL = "llama3.1:8b-instruct"  # 依你裝的模型調整
			 ```
		 - 或直接指定端點：
			 ```powershell
			 $env:OPENAI_BASE_URL = "http://localhost:11434/v1"
			 # 不需 API Key；程式會自動使用 dummy key
			 ```

- LM Studio（預設端點 http://localhost:1234/v1）
	1) 安裝並在「Server」分頁啟動 OpenAI 相容端點
	2) secrets（或環境變數）設定：
		 ```toml
		 LOCAL_LLM_PROVIDER = "lmstudio"
		 LOCAL_LLM_MODEL = "gpt-3.5-turbo"  # LM Studio 會映射到你選擇的本地模型
		 ```

LangChain 會：
1. 有 OPENAI_API_KEY → 用 ChatOpenAI。
2. 沒金鑰但有 LOCAL_LLM_PROVIDER=ollama → 用 ChatOllama。
3. 沒金鑰且 base_url 指向本地 → 嘗試 ChatOpenAI（dummy key）。
4. 以上都沒有 → 回到模擬回覆。\n\n注意：若同時沒有金鑰與本地端點，系統才會退回到簡易的模擬回覆。

---

## 版權

請勿上傳私密金鑰到版本控制；在本地以環境變數，雲端以 Secrets 管理。