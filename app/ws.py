"""① 即時 ASR WebSocket + 說話者辨識 + ② 定稿寫入儲存。

協定(維持與 App 相容):
  Client -> Server:
    binary                                      PCM16 LE mono 16k
    {"type":"config","diarization":bool,        開/關語者辨識、關聯會議(可選 speaker_count)
                     "meeting_id":str,"user_id":str}
    {"type":"end"}                              定稿 + (若有 meeting_id)寫入儲存 + 回 final
    {"type":"reset"}                            丟棄狀態重來
  Server -> Client:
    {"type":"partial",committed,tentative,text,diarization,segments}
    {"type":"final",text,segments,meeting_id}
    {"type":"config",diarization,meeting_id}
    {"type":"error",detail}
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config, db, models, rag
from app.auth import decode_jwt, tailscale_whois
from app import pyannote_diar
from app.diarize import SpeakerClusterer
from app.diarize import assign_all as diarize_assign_all

router = APIRouter()


@router.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    # --- 身分辨識:tailscale 模式取自 whois(來源IP);jwt 模式取自 token(?token=/header) ---
    if config.AUTH_MODE == "tailscale":
        _uid0 = await tailscale_whois(ws.client.host if ws.client else None)
    else:
        _auth = ws.headers.get("authorization", "")
        _tok = ws.query_params.get("token") or (
            _auth.split(" ", 1)[1] if _auth.lower().startswith("bearer ") else None)
        _uid0 = decode_jwt(_tok) if _tok else None
        if config.AUTH_REQUIRED and not _uid0:
            await ws.send_text(json.dumps({"type": "error", "detail": "需要登入"}, ensure_ascii=False))
            await ws.close(code=4401)
            return

    # --- 每連線狀態 ---
    pf_cache: dict = {}
    vad_cache: dict = {}
    audio_buf = np.zeros(0, dtype=np.float32)
    seg_samples: list = []
    seg_dur = 0.0
    segments: list = []                 # {"text","speaker","start_ms","end_ms"}
    cur_preview = ""
    pad_tail = np.zeros(0, dtype=np.float32)   # 上一段的尾巴,補到下一段前面當 lead-in
    diarize_on = config.DIARIZE_DEFAULT
    clusterer = SpeakerClusterer(config.SPK_THRESHOLD, config.SPK_PREFIX,
                                 config.SPK_MIN_NEW_SEC)
    spk_hist: list = []                 # [(segment idx, embedding, 真實秒數)] 供定期重分群
    spk_seen = 0                        # 上次重分群時的 spk_hist 長度
    sdiar = None                        # 串流 diarizer(None = 不用 pyannote)
    sdiar_cuts = False                  # True=由它決定切點;False=只提供標籤,切段仍走 VAD
    sdiar_idx: list = []                # (切段模式) diarizer 的 seq → segments 索引

    # ② 儲存關聯 / 時間軸
    meeting_id = None
    user_id = _uid0 or config.DEFAULT_USER
    elapsed_ms = 0.0                    # 整場已處理音訊(ms)
    seg_start_ms = 0.0                  # 目前這句開始時間
    seg_base = 0                        # 此會議 DB 既有段數(逐句寫入/reconnect-continue 的起點)
    active = False                      # 有進行中、尚未收尾的錄音(斷線時判斷要不要收尾)

    send_lock = asyncio.Lock()
    fin_queue: asyncio.Queue = asyncio.Queue()

    async def send(obj):
        async with send_lock:
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                pass

    def committed_str():
        if diarize_on:
            lines = []
            for s in segments:
                spk = s.get("speaker")
                t = models.to_tw(s["text"])
                lines.append(f"{spk}：{t}" if spk else t)
            return "\n".join(lines)
        return models.to_tw("".join(s["text"] for s in segments))

    def seg_list():
        return [{"speaker": s.get("speaker"), "text": models.to_tw(s["text"])} for s in segments]

    async def push_partial():
        committed = committed_str()
        tentative = models.to_tw(cur_preview)
        sep = "\n" if (diarize_on and committed and tentative) else ""
        await send({
            "type": "partial", "committed": committed, "tentative": tentative,
            "text": committed + sep + tentative, "language": config.ASR_LANG,
            "diarization": diarize_on, "segments": seg_list(),
        })

    async def finalizer():
        while True:
            item = await fin_queue.get()
            try:
                if item is None:
                    break
                idx, seg = item
                try:
                    try:
                        text = await models.finalize_qwen(seg)
                        if text:
                            segments[idx]["text"] = text
                    except Exception as e:
                        # 定稿逾時/失敗:保留 paraformer 預覽文字當退路,底下照樣寫入 DB。
                        # (以前這個例外會連帶跳過寫入 → 整句從逐字稿消失)
                        await send({"type": "error", "detail": f"finalize: {e}"})
                    # 標籤模式:先用 campplus 給一個**暫時**標籤,讓 speaker 不會是 null
                    # (App 不必為了「稍後才有語者」改任何東西),pyannote 約 15 秒後覆蓋掉。
                    if (diarize_on and seg.size and not sdiar_cuts
                            and segments[idx].get("speaker") is None):
                        try:
                            emb = await models.spk_embed(seg)
                            s0 = segments[idx]
                            dur = (s0["end_ms"] - s0["start_ms"]) / 1000.0   # 真實長度(不含 padding)
                            segments[idx]["speaker"] = clusterer.assign(emb, dur)
                            if sdiar is None:      # 有 pyannote 時由它負責修正,不必兩套互相蓋
                                spk_hist.append((idx, emb, dur))
                                await recluster_speakers()
                        except Exception as e:
                            await send({"type": "error", "detail": f"diarize: {e}"})
                    await push_partial()
                    # ★ 逐句立刻寫入 DB(不等 end):斷線也不丟已定稿的句子
                    s = segments[idx]
                    if meeting_id and (s.get("text") or "").strip():
                        try:
                            await db.upsert_segment(
                                meeting_id, seg_base + idx, models.to_tw(s["text"]),
                                s.get("speaker"), s.get("start_ms"), s.get("end_ms"))
                        except Exception as e:
                            await send({"type": "error", "detail": f"persist: {e}"})
                except Exception as e:
                    await send({"type": "error", "detail": f"segment: {e}"})
            finally:
                fin_queue.task_done()

    async def recluster_speakers(force: bool = False):
        """(B) 每累積 N 段就回頭全域重分群,修正先前判錯的語者標籤。

        線上貪婪是一次定案不可回頭:第 3 段判錯,後面所有證據都救不回它。這裡定期
        用全部已累積的聲紋重跑全域分群 → 改寫既有 segments 的標籤 → 同步更新 DB。
        WS 每次 partial 本來就重送整個 segments 陣列,App 會自動換成新標籤。
        重分群後把中心餵回 clusterer,讓後續線上判定與新編號一致。
        """
        every = config.SPK_RECLUSTER_EVERY
        nonlocal spk_seen
        if not spk_hist or (not force and (every <= 0 or len(spk_hist) - spk_seen < every)):
            return
        spk_seen = len(spk_hist)
        labels, cents, counts = diarize_assign_all(
            [e for _, e, _ in spk_hist], [d for _, _, d in spk_hist],
            config.SPK_THRESHOLD, config.SPK_PREFIX, config.SPK_MIN_NEW_SEC)
        clusterer.load_state(cents, counts)
        changed = []
        for (idx, _, _), lab in zip(spk_hist, labels):
            if segments[idx].get("speaker") != lab:
                segments[idx]["speaker"] = lab
                changed.append(idx)
        if not changed:
            return
        if meeting_id:
            for idx in changed:                      # 同步既有 DB 列的語者
                s = segments[idx]
                if (s.get("text") or "").strip():
                    try:
                        await db.upsert_segment(
                            meeting_id, seg_base + idx, models.to_tw(s["text"]),
                            s.get("speaker"), s.get("start_ms"), s.get("end_ms"))
                    except Exception as e:
                        await send({"type": "error", "detail": f"persist: {e}"})
        await push_partial()

    fin_task = asyncio.create_task(finalizer())

    def close_segment():
        nonlocal cur_preview, pf_cache, seg_samples, seg_dur, seg_start_ms, pad_tail
        seg = np.concatenate(seg_samples) if seg_samples else np.zeros(0, np.float32)
        idx = len(segments)
        segments.append({"text": cur_preview, "speaker": None,
                         "start_ms": int(seg_start_ms), "end_ms": int(elapsed_ms)})
        cur_preview = ""
        pf_cache = {}
        seg_samples = []
        seg_dur = 0.0
        seg_start_ms = elapsed_ms
        if seg.size == 0:
            return
        pad_n = int(config.SAMPLE_RATE * config.SEG_PAD_MS / 1000)
        if not models.is_speech_segment(seg):
            # 非語音(靜音/雜訊/碎段):不送定稿——送了會被幻覺成假句子,長靜音還會卡死佇列。
            # 連預覽文字一起清掉,才不會把 paraformer 對雜訊的臆測寫進 DB。
            segments[idx]["text"] = ""
            pad_tail = seg[-pad_n:] if pad_n else pad_tail
            return
        # 段前補前一段的尾巴當 lead-in(斷句處是靜音,不會重複字),讓模型不要從語音起點硬切進來
        fin_seg = np.concatenate([pad_tail, seg]) if pad_tail.size else seg
        pad_tail = seg[-pad_n:] if pad_n else pad_tail
        fin_queue.put_nowait((idx, fin_seg))

    async def drain_sdiar(chunk=None, final: bool = False):
        """語者切段模式:餵音訊 → 把「已定案」的段送去定稿 → 套用回頭修正的標籤。"""
        nonlocal cur_preview, pf_cache
        if sdiar is None:
            return
        try:
            new, relabels = await (sdiar.flush() if final
                                   else sdiar.push(chunk if chunk is not None
                                                   else np.zeros(0, np.float32)))
        except Exception as e:
            await send({"type": "error", "detail": f"diarize: {e}"})
            return
        for seq, lab in relabels.items():             # 重分群後標籤變了 → 改寫 + 同步 DB
            if seq < len(sdiar_idx):
                i = sdiar_idx[seq]
                segments[i]["speaker"] = lab
                s = segments[i]
                if meeting_id and (s.get("text") or "").strip():
                    try:
                        await db.upsert_segment(meeting_id, seg_base + i,
                                                models.to_tw(s["text"]), lab,
                                                s.get("start_ms"), s.get("end_ms"))
                    except Exception as e:
                        await send({"type": "error", "detail": f"persist: {e}"})
        if not sdiar_cuts:
            # 標籤模式:切段仍由 VAD 決定(字即時出來),這裡只用時間軸重貼所有段的標籤
            changed = []
            for i, s0 in enumerate(segments):
                lab = sdiar.label_for(s0.get("start_ms", 0), s0.get("end_ms", 0))
                if lab and s0.get("speaker") != lab:
                    s0["speaker"] = lab
                    changed.append(i)
            for i in changed:
                s0 = segments[i]
                if meeting_id and (s0.get("text") or "").strip():
                    try:
                        await db.upsert_segment(meeting_id, seg_base + i,
                                                models.to_tw(s0["text"]), s0["speaker"],
                                                s0.get("start_ms"), s0.get("end_ms"))
                    except Exception as e:
                        await send({"type": "error", "detail": f"persist: {e}"})
            if changed:
                await push_partial()
            return
        for row in new:
            idx = len(segments)
            segments.append({"text": "", "speaker": row["speaker"],
                             "start_ms": row["start_ms"], "end_ms": row["end_ms"]})
            sdiar_idx.append(idx)
            fin_queue.put_nowait((idx, row["audio"]))
        if new or relabels:
            if new:            # 這段文字已由定稿接手,預覽從頭開始(灰字只顯示尚未定案的部分)
                cur_preview, pf_cache = "", {}
            await push_partial()

    async def process_chunk(chunk, is_final):
        nonlocal cur_preview, seg_dur, elapsed_ms
        if chunk.size:
            seg_samples.append(chunk)
            seg_dur += chunk.size / config.SAMPLE_RATE
            elapsed_ms += chunk.size / config.SAMPLE_RATE * 1000

        if sdiar is not None:
            await drain_sdiar(chunk)      # 語者切段模式:由 diarizer 決定切點,不走 VAD

        seg_ended = False
        try:
            vad_res = await models.vad(chunk, vad_cache, is_final)
            if vad_res and vad_res[0].get("value"):
                for s in vad_res[0]["value"]:
                    if isinstance(s, (list, tuple)) and len(s) == 2 and s[1] != -1:
                        seg_ended = True
        except Exception:
            pass

        try:
            pf_res = await models.preview(chunk, pf_cache, is_final)
            if pf_res and pf_res[0].get("text"):
                cur_preview += pf_res[0]["text"]
        except Exception as e:
            await send({"type": "error", "detail": f"preview: {e}"})

        if seg_dur >= config.MAX_SEG_SEC:
            seg_ended = True
        if seg_ended and not is_final and not sdiar_cuts:   # 切段模式才由 diarizer 決定切點
            close_segment()
        await push_partial()

    def reset_all():
        nonlocal pf_cache, vad_cache, audio_buf, seg_samples, seg_dur
        nonlocal cur_preview, clusterer, elapsed_ms, seg_start_ms, pad_tail, spk_seen, sdiar, sdiar_cuts
        pf_cache = {}
        vad_cache = {}
        audio_buf = np.zeros(0, dtype=np.float32)
        seg_samples = []
        seg_dur = 0.0
        cur_preview = ""
        pad_tail = np.zeros(0, dtype=np.float32)
        segments.clear()
        clusterer = SpeakerClusterer(config.SPK_THRESHOLD, config.SPK_PREFIX,
                                     config.SPK_MIN_NEW_SEC)
        spk_hist.clear()
        spk_seen = 0
        sdiar = None
        sdiar_cuts = False
        sdiar_idx.clear()
        elapsed_ms = 0.0
        seg_start_ms = 0.0

    async def finalize_meeting():
        """收尾:設 status=ready + duration + 建向量索引。逐字稿已於 finalizer 逐句寫入,這裡不重寫。"""
        if not meeting_id:
            return
        try:
            await db.set_status(meeting_id, "ready", int(elapsed_ms / 1000))
        except Exception as e:
            await send({"type": "error", "detail": f"status: {e}"})
        try:
            await rag.index_meeting(user_id, meeting_id)   # ⑥ 建向量索引
        except Exception as e:
            await send({"type": "error", "detail": f"index: {e}"})

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                t = ctrl.get("type")
                if t == "config":
                    # jwt 模式才允許用 config 的 token/user_id 覆寫身分;tailscale 模式身分固定由 whois 決定
                    if config.AUTH_MODE == "jwt" and ctrl.get("token"):
                        _u = decode_jwt(ctrl["token"])
                        if _u:
                            user_id = _u
                    if ctrl.get("meeting_id"):
                        meeting_id = ctrl["meeting_id"]
                        seg_base = await db.count_segments(meeting_id)   # 續錄:接在既有段之後
                        active = True
                    if config.AUTH_MODE == "jwt" and ctrl.get("user_id") and not config.AUTH_REQUIRED:
                        user_id = ctrl["user_id"]              # 開發期才允許直接指定
                    if "diarization" in ctrl:
                        diarize_on = bool(ctrl["diarization"])
                        if diarize_on:
                            try:
                                await models.get_spk_model()
                            except Exception as e:
                                await send({"type": "error", "detail": f"diarize load: {e}"})
                    want_pya = config.WS_SEGMENT == "speaker" or \
                        config.WS_DIARIZE in ("auto", "pyannote")
                    if diarize_on and want_pya and sdiar is None:
                        try:
                            pipe = await pyannote_diar._get()
                            if pipe is not None:
                                from app.stream_diar import StreamingDiarizer
                                sdiar_cuts = config.WS_SEGMENT == "speaker"
                                sdiar = StreamingDiarizer(pipe, config.WS_DIAR_LATENCY,
                                                          config.WS_DIAR_RECLUSTER,
                                                          emit=sdiar_cuts)
                                print(f"[ws] pyannote 啟用:"
                                      f"{'依語者切段(定稿慢約 20 秒)' if sdiar_cuts else '只貼標籤(字即時,標籤晚約 15 秒)'}")
                        except Exception as e:
                            await send({"type": "error", "detail": f"diarize load: {e}"})
                    if meeting_id:
                        try:
                            await db.set_status(meeting_id, "transcribing")
                        except Exception:
                            pass
                    await send({"type": "config", "diarization": diarize_on,
                                "meeting_id": meeting_id})
                    continue

                if t == "end":
                    if audio_buf.size > 0:
                        await process_chunk(audio_buf, is_final=True)
                        audio_buf = np.zeros(0, dtype=np.float32)
                    if not sdiar_cuts:
                        close_segment()
                    if sdiar is not None:
                        await drain_sdiar(final=True)
                    await fin_queue.join()      # 等最後一句定稿 + 寫入 DB
                    await recluster_speakers(force=True)   # 收尾:用全部聲紋做最後一次修正
                    await finalize_meeting()    # 設 ready + 索引(逐字稿早已逐句寫好)
                    await send({"type": "final", "text": committed_str(),
                                "segments": seg_list(), "meeting_id": meeting_id})
                    seg_base += len(segments)   # 推進起點(同連線若繼續錄下一段)
                    active = False
                    reset_all()
                elif t == "reset":
                    reset_all()
                continue

            data = msg.get("bytes")
            if not data:
                continue
            seg = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if seg.size == 0:
                continue
            audio_buf = np.concatenate([audio_buf, seg])
            while audio_buf.size >= config.CHUNK_STRIDE:
                chunk = audio_buf[:config.CHUNK_STRIDE]
                audio_buf = audio_buf[config.CHUNK_STRIDE:]
                await process_chunk(chunk, is_final=False)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await send({"type": "error", "detail": str(e)})
    finally:
        # 斷線(沒收到 end)也 best-effort 收尾:收掉目前這句、等佇列寫完 → 已錄的都進 DB
        if active:
            try:
                if seg_samples or cur_preview:
                    close_segment()
                await asyncio.wait_for(fin_queue.join(), timeout=8)
            except Exception:
                pass
        try:
            fin_queue.put_nowait(None)
            await asyncio.wait_for(fin_task, timeout=5)
        except Exception:
            fin_task.cancel()
        # 中途斷線 → 標記 ready + 索引,不讓會議卡在「轉錄中」
        if meeting_id and active:
            try:
                await db.set_status(meeting_id, "ready", int(elapsed_ms / 1000))
                await rag.index_meeting(user_id, meeting_id)
            except Exception:
                pass
