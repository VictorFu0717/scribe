"""把會議內容畫成圖(⑦ agent 工具 draw_meeting_diagram 用)。

做法:讓對話 LLM 產生 **Mermaid** 原始碼,由 App 端渲染。
為什麼不在 server 端算成圖片:
  - Mermaid 是純文字,LLM 直接生得出來,不必在推論路徑上再掛一個瀏覽器
  - App 端渲染才能縮放/點選/複製,出圖也比較好看
  - 真要圖片,之後在這層後面補一個 renderer 即可,工具契約不用動

輸入優先用「摘要」:它已經是整理過的濃縮版,畫流程圖比逐字稿準得多。
沒有摘要才退回逐字稿(截斷到 DIAGRAM_MAX_CHARS)。
"""

from __future__ import annotations

import json
import re

from app import config, db, llm

# Mermaid 圖種 → (標頭關鍵字, 給 LLM 的指示)
KINDS = {
    "flow": ("flowchart", "流程圖:把會議談到的流程/步驟/決策分支畫成 `flowchart TD`。"
                          "決策點用菱形 `id{\"問題?\"}`,分支邊標上條件。"),
    "relation": ("graph", "關係圖:把人、單位、系統、專案之間的關係畫成 `graph LR`。"
                          "邊上標明關係(負責、依賴、提供、回報)。"),
    "timeline": ("timeline", "時間軸:把會議談到的里程碑/期程畫成 `timeline`。"
                             "每個 section 是一個階段,底下列事件。"),
    "mindmap": ("mindmap", "心智圖:把會議主題階層化畫成 `mindmap`,根節點是會議主題。"),
}

_SYS = """你是會議圖表產生器。根據提供的會議內容,輸出一張 Mermaid 圖。

{kind_hint}

硬性規則:
- 只輸出 Mermaid 原始碼,不要加說明、不要包 ```mermaid 圍欄。
- 節點文字一律用雙引號包起來:`a["文字"]`。中文標點(、,。:)放在引號內才安全。
- 節點 id 只能用英數與底線(n1, n2 ...),**不要**用中文當 id。
- 節點數控制在 4~12 個,挑重點,不要把逐字稿每句都畫進去。
- 一律使用繁體中文(台灣用語)。
- 內容不足以畫圖時,輸出單一節點說明原因,不要編造。"""

_FENCE = re.compile(r"^\s*```(?:mermaid)?\s*|\s*```\s*$", re.M)


def _clean(raw: str) -> str:
    """剝掉 LLM 常加的 ``` 圍棚與前後空白。"""
    return _FENCE.sub("", raw or "").strip()


def validate(code: str, kind: str) -> str | None:
    """回傳錯誤描述;通過則回 None。擋掉 LLM 最常犯的幾種語法錯。"""
    if not code:
        return "輸出是空的"
    head = KINDS[kind][0]
    first = code.splitlines()[0].strip() if code.splitlines() else ""
    if not first.startswith(head):
        return f"第一行必須以 `{head}` 開頭,實際是:{first[:40]!r}"
    if code.count("[") != code.count("]"):
        return "方括號不成對"
    if code.count("{") != code.count("}"):
        return "大括號不成對"
    if code.count('"') % 2:
        return "雙引號不成對"
    if re.search(r"(?<![\w\"])[一-鿿]+\s*[\[\{\(]", code):
        return "有節點用中文當 id(中文必須放在引號內的標籤,不能當 id)"
    if len(code.splitlines()) < 2:
        return "只有一行,沒有任何節點或關係"
    return None


async def build(user_id: str, meeting_id: str, kind: str = "flow",
                focus: str | None = None) -> dict:
    """產生一張圖。回傳 {"kind","code","source"};失敗時丟例外由呼叫端處理。"""
    kind = kind if kind in KINDS else "flow"
    if await db.get_meeting(user_id, meeting_id) is None:
        raise ValueError("找不到該會議")

    sm = await db.get_summary(user_id, meeting_id)
    if sm:
        body = json.dumps(sm, ensure_ascii=False, indent=1)
        src = "summary"
    else:
        body = (await db.get_transcript_text(meeting_id))[:config.DIAGRAM_MAX_CHARS]
        src = "transcript"
    if not body.strip():
        raise ValueError("此會議沒有可用內容")

    user = f"會議內容:\n{body}"
    if focus:
        user += f"\n\n請特別聚焦在:{focus}"

    msgs = [{"role": "system", "content": _SYS.format(kind_hint=KINDS[kind][1])},
            {"role": "user", "content": user}]
    code = _clean(await llm.once(msgs, temperature=0.2))
    err = validate(code, kind)
    if err:   # 給 LLM 一次自我修正的機會(實測多半是圍欄或中文 id)
        print(f"[diagram] {meeting_id} 第一次語法有誤({err}),重試一次")
        msgs += [{"role": "assistant", "content": code},
                 {"role": "user", "content": f"這段 Mermaid 有錯:{err}。請重新輸出修正後的完整原始碼。"}]
        code = _clean(await llm.once(msgs, temperature=0.0))
        err = validate(code, kind)
        if err:
            raise ValueError(f"產生的 Mermaid 語法有誤:{err}")
    out = {"kind": kind, "code": code, "source": src}
    if src == "transcript":
        # 逐字稿是口語、細節散落,畫出來比摘要差一截。明講出來,讓使用者知道還有更好的選項,
        # 而不是以為「這個功能就只能畫成這樣」。
        out["hint"] = "這場會議還沒有摘要,這張圖是直接讀逐字稿畫的。產生摘要後重畫會更準確。"
    return out
