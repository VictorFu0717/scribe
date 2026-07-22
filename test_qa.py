"""測試 ③ 單場會議 QA:POST /meeting/chat,串流印出答案。

前置:
    1) 起 chat LLM:  cd ~/PycharmProjects/RAG_LangChain/vllm && docker compose up -d   (:8004)
    2) 起 scribe:    python main.py                                                     (:8005)
    3) 跑本檔:       python test_qa.py
"""
import json
import httpx

SCRIBE = "http://localhost:8005/meeting/chat"

TRANSCRIPT = (
    "王經理:這季營收成長了百分之十五,主要來自新客戶。"
    "李副理:好,那行銷預算下一季要不要加碼?"
    "王經理:先加兩成,重點放在數位廣告。"
    "李副理:了解,我下週五前把預算表和這季報表整理給你。"
    "王經理:另外提醒大家,月底前要完成年度考核表。"
)

QUESTIONS = [
    "這場會議有哪些待辦事項?各自的負責人與期限?",
    "這季營收表現如何?",
    "會議有討論到放假嗎?",   # 逐字稿沒有 → 應回答「沒有提到」
]


def ask(question: str, history: list[dict]) -> str:
    body = {"transcript": TRANSCRIPT, "question": question, "history": history}
    answer = ""
    with httpx.stream("POST", SCRIBE, json=body, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "error" in obj:
                return f"[error] {obj['error']}"
            piece = obj.get("delta", "")
            answer += piece
            print(piece, end="", flush=True)
    print()
    return answer


def main():
    history: list[dict] = []
    for q in QUESTIONS:
        print(f"\n\033[1m❓ {q}\033[0m")
        print("💬 ", end="")
        a = ask(q, history)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": a})


if __name__ == "__main__":
    main()
