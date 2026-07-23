"""⑤ agentic 個人助理 — POST /assistant/chat (SSE 串流)。

手寫 agent loop:LLM 拿到工具清單,自己決定要不要呼叫(多輪),最後串流答案。
工具註冊表好擴充(加工具 = 加一個 schema + handler)。目前工具:
    get_meeting_transcript / get_meeting_summary / list_meetings / search_meetings(關鍵字,⑥ 升級語意)

body: {"messages":[{"role","content"}...], "meeting_id":str|null, "language":"zh-Hant"}
  - 有 meeting_id → 提示 agent 以該場為「目前會議」;無 → 可跨會議(search/list)。
回傳(SSE): data:{"delta":"..."} ... data:[DONE]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config, db, llm, rag
from app.auth import get_current_user

router = APIRouter(tags=["assistant"])

MAX_STEPS = 5   # 工具呼叫輪數上限(防迴圈)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_meeting_transcript",
        "description": "取得指定會議的完整逐字稿(含說話者)。回答某場會議內容時用。",
        "parameters": {"type": "object", "properties": {
            "meeting_id": {"type": "string", "description": "會議 id"}}, "required": ["meeting_id"]}}},
    {"type": "function", "function": {
        "name": "get_meeting_summary",
        "description": "取得指定會議的結構化摘要(重點/決議/待辦)。",
        "parameters": {"type": "object", "properties": {
            "meeting_id": {"type": "string"}}, "required": ["meeting_id"]}}},
    {"type": "function", "function": {
        "name": "list_meetings",
        "description": "列出使用者所有會議(id/標題/時間/狀態)。需要先找出是哪場會議時用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_meetings",
        "description": "跨所有會議做語意搜尋逐字稿,回傳相關片段與所屬會議。回答跨會議問題"
                       "(如某主題、待辦、某段時間的重點)時用。可用日期範圍限定。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要找的內容(語意查詢)"},
            "date_from": {"type": "string", "description": "起始日期 YYYY-MM-DD(可選)"},
            "date_to": {"type": "string", "description": "結束日期 YYYY-MM-DD(可選)"}},
            "required": ["query"]}}},
]

SYSTEM = (
    "你是使用者的個人會議助理。你可以呼叫工具查詢使用者的會議逐字稿與摘要來回答問題。\n"
    "規則:\n"
    "- 一律使用繁體中文(台灣用語)。\n"
    "- 根據工具查到的內容回答;查不到就說「找不到相關資訊」,不要編造。\n"
    "- 需要某場會議內容時,先用 list_meetings/search_meetings 找出 meeting_id,再取逐字稿或摘要。\n"
    "- 若某場會議「摘要尚未產生」,改用 get_meeting_transcript 取逐字稿來回答,不要因為沒有摘要就放棄。\n"
    "- 回答精簡、切中重點,必要時條列。"
)


async def _run_tool(name: str, args_str: str, user: str) -> str:
    try:
        args = json.loads(args_str or "{}")
    except Exception:
        args = {}
    if name == "get_meeting_transcript":
        mid = args.get("meeting_id", "")
        if not await db.get_meeting(user, mid):
            return "找不到該會議"
        return (await db.get_transcript_text(mid)) or "(此會議尚無逐字稿)"
    if name == "get_meeting_summary":
        mid = args.get("meeting_id", "")
        s = await db.get_summary(user, mid)
        return json.dumps(s, ensure_ascii=False) if s else "(此會議尚無摘要)"
    if name == "list_meetings":
        ms = await db.list_meetings(user)
        return json.dumps([{"id": m["id"], "title": m["title"],
                            "created_at": m["created_at"], "status": m["status"]}
                           for m in ms], ensure_ascii=False)
    if name == "search_meetings":
        try:
            rows = await rag.semantic_search(user, args.get("query", ""),
                                             date_from=args.get("date_from"),
                                             date_to=args.get("date_to"))
        except Exception:
            rows = await db.search_transcripts(user, args.get("query", ""))   # 退回關鍵字
        rows = [{k: v for k, v in r.items() if k != "distance"} for r in rows]
        return json.dumps(rows, ensure_ascii=False) if rows else "(找不到相關會議)"
    return f"未知工具:{name}"


class AssistantReq(BaseModel):
    messages: list[dict]
    meeting_id: str | None = None
    language: str | None = "zh-Hant"


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/assistant/chat")
async def assistant_chat(req: AssistantReq, user: str = Depends(get_current_user)):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sys = SYSTEM + (f"\n\n今天是 {today}。若使用者用相對時間(如「上週」「上個月5號」),"
                    f"請自行換算成 date_from/date_to(YYYY-MM-DD)傳給 search_meetings。")
    if req.meeting_id:
        sys += (f"\n\n使用者目前正在看會議 id=「{req.meeting_id}」;"
                f"問「這場/本次會議」相關問題時,以此會議為準(可直接對它取逐字稿/摘要)。")
    messages = [{"role": "system", "content": sys}]
    messages += [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in req.messages]

    async def gen():
        try:
            for _ in range(MAX_STEPS):
                stream = await llm.client.chat.completions.create(
                    model=config.CHAT_MODEL, messages=messages, tools=TOOLS,
                    stream=True, temperature=0.3,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                content = ""
                acc: dict = {}   # index -> {id,name,args}
                async for ch in stream:
                    if not ch.choices:
                        continue
                    d = ch.choices[0].delta
                    if d.content:
                        content += d.content
                        yield _sse({"delta": d.content})
                    if d.tool_calls:
                        for tc in d.tool_calls:
                            e = acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id:
                                e["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    e["name"] += tc.function.name
                                if tc.function.arguments:
                                    e["args"] += tc.function.arguments

                if not acc:
                    return   # 沒有工具呼叫 → content 就是最終答案(已串流)

                # 記錄 assistant 的 tool_calls,再逐一執行、把結果餵回
                messages.append({
                    "role": "assistant", "content": content or None,
                    "tool_calls": [{"id": e["id"], "type": "function",
                                    "function": {"name": e["name"], "arguments": e["args"]}}
                                   for e in acc.values()]})
                for e in acc.values():
                    result = await _run_tool(e["name"], e["args"], user)
                    messages.append({"role": "tool", "tool_call_id": e["id"], "content": result})
            # 迴圈用盡仍沒收斂
            yield _sse({"delta": "\n(已達工具呼叫上限)"})
        except Exception as ex:
            yield _sse({"error": str(ex)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
