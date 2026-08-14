# app/chat.py
import os
import json
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


from .morph import MorphService
from .memory import apply_explicit_memory

def _shared():
    from . import main as _m
    return _m._state_path, _m._ensure_cache, _m.call_llm

router = APIRouter()

morph = MorphService()

class ChatIn(BaseModel):
    intent: Optional[str] = None   # quiz3 | confuse | examples2 | family5 | regen_explain | regen_quiz
    message: Optional[str] = None  # 纯聊天时使用；有 intent 时可为空
    force_new: bool = False        # True 则清空该词历史再聊

# ---------- Helpers ----------
#去morph找单词的基本信息
def _grounding_for(word: str) -> Dict[str, Any]:
    ex = morph.analyze(word, lang=None) or {}
    # merge various possible keys
    group_label = ex.get("root") or ex.get("affix") or (ex.get("group", {}) or {}).get("label") or ""
    neighbors = ex.get("neighbors") or (ex.get("group", {}) or {}).get("neighbors") or []
    examples = ex.get("examples") or ex.get("example_sentences") or ex.get("example_sents") or []
    collos = ex.get("collocations") or ex.get("collocation_hints") or []
    forms = ex.get("forms") or ex.get("derived_forms") or ex.get("inflections") or []
    # try method-based fallbacks if available
    try:
        if not examples and hasattr(morph, "get_examples"):
            examples = morph.get_examples(word) or []
    except Exception:
        pass
    try:
        if not collos and hasattr(morph, "get_collocation_hints"):
            collos = morph.get_collocation_hints(word) or []
    except Exception:
        pass
    return {
        "word": word,
        "decomposition": ex.get("decomposition", ""),
        "group": {"label": group_label, "neighbors": neighbors},
        "collocations": collos,
        "examples": examples,
        "forms": forms,
    }
async def _llm_plain(system_prompt: str, user_payload: Dict[str, Any]) -> str:
    """
    Call LLM and return plain text (no JSON). Works with sync/async call_llm.
    """
    _state_path, _ensure_cache, _call_llm = _shared()
    prompt = (
        system_prompt.strip()
        + "\n\nUSER_PAYLOAD:\n"
        + json.dumps(user_payload, ensure_ascii=False)
        + "\n\nReturn ONLY plain text. No JSON. No code fences."
    )
    _resp = _call_llm(prompt)
    import inspect
    if inspect.isawaitable(_resp):
        return (await _resp) or ""
    return _resp or ""

async def _llm_json(system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the shared call_llm while being compatible with both sync/async implementations.
    We do NOT pass unsupported keyword args; instead we combine system + user into one prompt.
    We also robustly strip code fences and parse JSON.
    """
    _state_path, _ensure_cache, _call_llm = _shared()

    # Compose a single prompt
    prompt = (
        system_prompt.strip()
        + "\n\nUSER_PAYLOAD:\n"
        + json.dumps(user_payload, ensure_ascii=False)
        + "\n\nReturn ONLY a compact JSON object. Do NOT wrap in code fences. No commentary."
    )

    # Call and support both async/sync versions
    _resp = _call_llm(prompt)
    import inspect
    if inspect.isawaitable(_resp):
        txt = await _resp
    else:
        txt = _resp

    if not txt:
        return {}

    s = str(txt).strip()
    # Strip optional ```json fences
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # Try JSON parse, with fallback to first {...} or [...] block
    try:
        return json.loads(s)
    except Exception:
        import re
        m = re.search(r"([\\[{].*[\\]}])", s, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return {}

def _format_quiz(payload: Dict[str, Any]) -> str:
    items: List[Dict[str, Any]] = payload.get("items") or payload.get("quiz") or []
    lines = [" Quiz (3 items)"]
    for i, it in enumerate(items, 1):
        t = (it.get("type") or "").upper()
        stem = it.get("stem") or it.get("question") or ""
        ans = it.get("answer")
        if isinstance(ans, list):
            ans = ", ".join(map(str, ans))
        opts = it.get("options")
        lines.append(f"{i}) [{t}] {stem}")
        if opts:
            lines.append("   options: " + ", ".join(map(str, opts)))
        if ans:
            lines.append(f"   answer: {ans}")
    return "\n".join(lines)

def _format_confuse(arr: Any) -> str:
    # Normalize items from various possible keys/structures
    if isinstance(arr, dict):
        if "pairs" in arr:
            items = arr["pairs"]
        elif "confusables" in arr:
            items = arr["confusables"]
        elif "items" in arr:
            items = arr["items"]
        elif "data" in arr:
            items = arr["data"]
        elif "result" in arr:
            items = arr["result"]
        else:
            items = []
    elif isinstance(arr, list):
        items = arr
    else:
        items = []

    lines = [" Confusables"]
    for it in items:
        if not isinstance(it, dict):
            lines.append(f"- {it}")
            continue
        w = it.get("word") or it.get("vs") or it.get("term") or it.get("target") or "?"
        diff = it.get("difference") or it.get("diff") or it.get("explain") or it.get("note") or ""
        coll = it.get("example_collocation") or it.get("collocation") or it.get("pattern") or it.get("example") or ""
        line = f"- vs {w}: {diff}".strip()
        if coll:
            line += f" | e.g., {coll}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("- (no items)")
    return "\n".join(lines)

def _format_examples(payload: Dict[str, Any]) -> str:
    exs = (
        payload.get("examples")
        or payload.get("items")
        or payload.get("data")
        or payload.get("result")
        or []
    )
    lines = ["Examples"]
    for e in exs:
        if isinstance(e, str):
            lines.append(f"- {e}")
        elif isinstance(e, dict):
            s = e.get("sentence") or e.get("en") or e.get("text") or ""
            if s:
                lines.append(f"- {s}")
    if len(lines) == 1:
        lines.append("- (no items)")
    return "\n".join(lines)

def _format_family(payload: Dict[str, Any]) -> str:
    fam = (
        payload.get("family")
        or payload.get("family_words")
        or payload.get("word_family")
        or payload.get("forms")
        or payload.get("items")
        or payload.get("data")
        or payload.get("result")
        or []
    )
    lines = ["Word Family"]
    for it in fam:
        if isinstance(it, dict):
            form = it.get("form") or it.get("word") or it.get("lemma") or it.get("derived") or ""
            gloss = it.get("gloss") or it.get("hint") or it.get("meaning") or it.get("note") or ""
            entry = f"- {form}: {gloss}".rstrip(": ")
            lines.append(entry)
        else:
            lines.append(f"- {it}")
    if len(lines) == 1:
        lines.append("- (no items)")
    return "\n".join(lines)

@router.post("/api/vocab/{book_id}/{word}/chat")
async def vocab_chat(book_id: str, word: str, payload: ChatIn):
    _state_path, _ensure_cache, _call_llm = _shared()
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")

    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)
    vocab_cache = cache["vocab"]
    wkey = word.strip().lower()

    # Conversation state per word
    conv = vocab_cache.setdefault(wkey, {}).setdefault("chat", [])
    if payload.force_new:
        conv.clear()

    intent = (payload.intent or "").strip().lower()
    print("DEBUG intent:", repr(intent), "msg?=", bool(payload.message))

    # If we have an intent, route to structured LLM flows (JSON-returning)
    if intent in {"quiz3", "confuse", "examples2", "family5", "regen_explain", "regen_quiz"}:
        g = _grounding_for(word)

        if intent in {"quiz3", "regen_quiz"}:
            sys = (
                "You are a concise ESL quiz generator. Output JSON only. "
                "3 items: 1 cloze, 1 MCQ (3 options), 1 collocation. Level A2–B1. "
                "Avoid rare meanings. Keep stems ≤ 12 words."
            )
            user = {"word": word, "grounding": g, "format": {
                "items":[
                    {"type":"cloze","stem":"","answer":""},
                    {"type":"mcq","stem":"","options":[],"answer":""},
                    {"type":"collocation","stem":"","answer":""}
                ]},
                "avoid": []
            }
            data = await _llm_json(sys, user)
            assistant_msg = _format_quiz(data) if data else "Quiz generation failed."

        elif intent == "confuse":
            sys = (
                "You are a lexicographer. Output JSON only. Compare with 2 confusables. "
                "Each item must have: \"word\", \"difference\" (<=20 words), \"example_collocation\"."
            )
            user = {"word": word, "grounding": g}
            data = await _llm_json(sys, user)
            assistant_msg = _format_confuse(data) if data else "Confusables generation failed."

        elif intent == "examples2":
            sys = (
                "You are an ESL example writer. Output JSON only. "
                "Two A2-level sentences using the target word in common senses. Daily style."
            )
            user = {"word": word, "grounding": g}
            data = await _llm_json(sys, user)
            assistant_msg = _format_examples(data) if data else "Examples generation failed."
            # fallback if no examples or only header or ends with (no items)
            if not assistant_msg or assistant_msg.strip() == "Examples" or assistant_msg.strip().endswith("(no items)"):
                fallback_lines: List[str] = []

                # 1) grounding examples
                for e in (g.get("examples") or []):
                    s = ""
                    if isinstance(e, dict):
                        s = e.get("sentence") or e.get("en") or e.get("text") or ""
                    elif isinstance(e, str):
                        s = e
                    s = (s or "").strip()
                    if s:
                        fallback_lines.append(s)
                    if len(fallback_lines) >= 2:
                        break

                # 2) use collocations/collocation_hints to synthesize
                if len(fallback_lines) < 2:
                    for c in (g.get("collocations") or []):
                        ctext = ""
                        if isinstance(c, str):
                            ctext = c
                        elif isinstance(c, dict):
                            ctext = c.get("example") or c.get("phrase") or c.get("text") or ""
                        ctext = (ctext or "").strip()
                        if not ctext:
                            continue
                        # simple natural sentence from collocation
                        if " " in ctext:
                            fallback_lines.append(f"I often use '{ctext}' in daily life.")
                        else:
                            fallback_lines.append(f"This word appears in the collocation '{ctext}'.")
                        if len(fallback_lines) >= 2:
                            break

                # 3) final resort: ask LLM for plain text sentences
                if len(fallback_lines) < 2:
                    plain = await _llm_plain(
                        "Write two short A2-level example sentences using the target word. Return only two lines starting with '- '.",
                        {"word": word}
                    )
                    for line in (plain or "").splitlines():
                        t = line.strip().lstrip("-").strip()
                        if t:
                            fallback_lines.append(t)
                        if len(fallback_lines) >= 2:
                            break

                assistant_msg = "Examples\n" + ("\n".join(f"- {s}" for s in fallback_lines[:2]) if fallback_lines else "- (no items)")

        elif intent == "family5":
            sys = (
                "You are a morphology coach. Output JSON only. "
                "Up to 5 frequent family/derived forms with a 1-3 word gloss each."
            )
            user = {"word": word, "grounding": g}
            data = await _llm_json(sys, user)
            assistant_msg = _format_family(data) if data else "Word family generation failed."
            # fallback if no items or only header or ends with (no items)
            if not assistant_msg or assistant_msg.strip() == "Word Family" or assistant_msg.strip().endswith("(no items)"):
                pairs: List[tuple] = []

                # 1) neighbors
                for n in ((g.get("group") or {}).get("neighbors") or []):
                    if isinstance(n, dict):
                        w2 = (n.get("form") or n.get("word") or n.get("lemma") or n.get("derived") or "").strip()
                        g2 = (n.get("gloss") or n.get("hint") or n.get("meaning") or n.get("note") or "").strip()
                    else:
                        w2, g2 = str(n).strip(), ""
                    if w2:
                        pairs.append((w2, g2))
                    if len(pairs) >= 5:
                        break

                # 2) forms
                if len(pairs) < 5:
                    for f in (g.get("forms") or []):
                        if isinstance(f, dict):
                            w2 = (f.get("form") or f.get("word") or f.get("lemma") or f.get("derived") or "").strip()
                            g2 = (f.get("gloss") or f.get("hint") or f.get("meaning") or f.get("note") or "").strip()
                        else:
                            w2, g2 = str(f).strip(), ""
                        if w2:
                            pairs.append((w2, g2))
                        if len(pairs) >= 5:
                            break

                # 3) final resort: plain text list
                if len(pairs) < 3:
                    plain = await _llm_plain(
                        "List up to 5 frequent family/derived forms of the target word. One per line in the format 'word — very short gloss'.",
                        {"word": word}
                    )
                    for line in (plain or "").splitlines():
                        t = (line or "").strip()
                        if not t:
                            continue
                        if "—" in t:
                            w2, g2 = [x.strip() for x in t.split("—", 1)]
                        elif "-" in t:
                            w2, g2 = [x.strip() for x in t.split("-", 1)]
                        else:
                            w2, g2 = t, ""
                        if w2:
                            pairs.append((w2, g2))
                        if len(pairs) >= 5:
                            break

                # assemble
                if pairs:
                    lines = ["Word Family"]
                    for w2, g2 in pairs[:5]:
                        lines.append(f"- {w2}" + (f": {g2}" if g2 else ""))
                    assistant_msg = "\n".join(lines)
                else:
                    assistant_msg = "Word Family\n- (no items)"

        elif intent == "regen_explain":
            sys = (
                "Regenerate a 5-part explanation. Output JSON keys: decomposition, group, mnemonic, confusables, usage. "
                "Keep concise. Respect the provided grounding facts."
            )
            user = {"word": word, "grounding": g, "style": "simpler"}
            data = await _llm_json(sys, user)
            # For now, return JSON as text; front-end could later render a card
            assistant_msg = json.dumps(data, ensure_ascii=False, indent=2) if data else "Regeneration failed."

        # Append to conversation and persist
        conv.append({"role": "user", "content": payload.message or f"[{intent}]"})
        conv.append({"role": "assistant", "content": assistant_msg})
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"word": word, "messages": conv[-20:]}

    # ---- Default plain chat (no intent) ----
    user_text = payload.message or ""
    if not user_text:
        raise HTTPException(400, "message is required for plain chat")

    # Durable memory updates are conservative and deterministic. They happen on
    # the same in-memory state object as chat persistence, avoiding a second
    # writer that could be overwritten by the final state-file write.
    apply_explicit_memory(state, user_text)
    conv.append({"role": "user", "content": user_text})

    try:
        # Keep this import inside the guarded path so a LangGraph import or
        # initialization failure can still use the existing chat implementation.
        from .agent import run_agent

        ai_text = await run_agent(
            book_id=book_id,
            active_word=word,
            conversation=conv,
            memory=state.get("memory"),
        )
    except Exception as e:
        print(f"Agent failure; using legacy plain-chat fallback: {type(e).__name__}: {e}")
        sys_prompt = (
            f"You are an English vocabulary coach. Current word: {word}. "
            "Explain step-by-step using a root-and-suffix approach, "
            "provide concise and clear answers, and pose a brief follow-up question to confirm understanding. "
            "Output plain text only."
        )
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in conv)
        ai_text = await _call_llm(sys_prompt + "\n\n" + history_text + "\n\nassistant:")
    assistant_msg = (ai_text or "").strip() or "(Placeholder response: No actual model received)"

    conv.append({"role": "assistant", "content": assistant_msg})
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"word": word, "messages": conv[-20:]}
#取历史借口
@router.get("/api/vocab/{book_id}/{word}/chat/history")
async def chat_history(book_id: str, word: str):
    _state_path, _ensure_cache, _ = _shared()
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)

    wkey = word.strip().lower()
    conv = cache["vocab"].get(wkey, {}).get("chat", [])
    return {"word": word, "messages": conv[-100:]}
