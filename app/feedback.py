# app/feedback.py
import json, os, inspect
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException

from .morph import MorphService

router = APIRouter()
morph = MorphService()

# 复用 main 里共享工具
def _shared():
    from . import main as _m
    return _m._state_path, _m._ensure_cache, _m.call_llm

# 载入反馈库（带多路径兜底 + 调试打印）
_FEEDBACK = []
def _load_feedback() -> list:
    global _FEEDBACK
    if _FEEDBACK:
        return _FEEDBACK

    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "data", "feedback.json"),  # app/data/feedback.json
        os.path.join(base, "feedback.json"),  # app/feedback.json
        os.path.join(base, "..", "data", "feedback.json"),  # 项目根目录的 data/feedback.json
        os.path.join(os.getcwd(), "app", "data", "feedback.json"),
        os.path.join(os.getcwd(), "data", "feedback.json"),  # 以防工作目录不同
    ]
    print("DEBUG feedback lookup candidates:", candidates)

    for p in candidates:
        if os.path.exists(p):
            print("DEBUG feedback loaded from:", p)
            with open(p, "r", encoding="utf-8") as f:
                _FEEDBACK = json.load(f)
            return _FEEDBACK

    # 清晰报错，直接把尝试过的路径打出来
    raise HTTPException(500, f"feedback.json not found. Tried: {candidates}")

def _grounding(word: str) -> Dict[str, Any]:
    ex = morph.analyze(word, lang=None) or {}
    return {
        "word": word,
        "decomposition": ex.get("decomposition", ""),
        "group": {"label": ex.get("root") or ex.get("affix") or ""},
        "collocations": ex.get("collocations", []),
        "examples": ex.get("examples", []),
    }

def _recent_chat(state: Dict[str, Any], word: str, limit: int = 8) -> List[Dict[str, str]]:
    wkey = word.strip().lower()
    msgs = state.get("vocab", {}).get(wkey, {}).get("chat", [])
    # 只取最近几条且裁剪文本
    out = []
    for m in msgs[-limit:]:
        out.append({"role": m.get("role",""), "text": str(m.get("content",""))[:200]})
    return out

def _search_feedback(word: str, tags: List[str], topk: int = 8) -> List[Dict[str, Any]]:
    items = _load_feedback()
    word_l = word.lower()
    scored = []
    for it in items:
        s = 0
        if it.get("word"):
            if it["word"].lower() == word_l: s += 3
        if it.get("tag"):
            it_tag = str(it.get("tag","")).lower()
            for t in tags:
                t_low = str(t).lower().strip()
                if t_low in it_tag: s += 2
        # 一些通用规则也给 1 分
        if not it.get("word"):
            s += 1
        if s>0:
            scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [it for _, it in scored[:topk]]
    if not results:
        # fallback: return some generic feedback items
        results = [
            {"tag": "generic", "detail": "Keep practicing this word in different contexts."},
            {"tag": "generic", "detail": "Review collocations and example sentences to strengthen usage."}
        ]
    return results

async def _llm_json(system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
    # 复用主程序 call_llm；兼容同步/异步；不传不支持的 kwargs
    _state_path, _ensure_cache, call_llm = _shared()
    prompt = (
        system_prompt.strip() + "\n\nUSER_PAYLOAD:\n" +
        json.dumps(user_payload, ensure_ascii=False) +
        "\n\nReturn ONLY a compact JSON object. No code fences. No commentary."
    )
    resp = call_llm(prompt)
    if inspect.isawaitable(resp):
        txt = await resp
    else:
        txt = resp or ""

    s = str(txt).strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()

    try:
        return json.loads(s)
    except Exception:
        # 简单兜底：抓第一个大括号或中括号块
        import re
        m = re.search(r"([\\[{].*[\\]}])", s, re.S)
        if m:
            try: return json.loads(m.group(1))
            except: pass
        return {}

def _uniq(lst: List[Any]) -> List[Any]:
    seen, res = set(), []
    for s in lst or []:
        k = json.dumps(s, ensure_ascii=False, sort_keys=True)
        if k not in seen:
            seen.add(k); res.append(s)
    return res

def _empty_block(d: Dict[str, Any]) -> bool:
    """True if strengths/issues/tips are all empty/missing."""
    if not d:
        return True
    return not ((d.get("strengths") or []) or (d.get("issues") or []) or (d.get("tips") or []))

def _build_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge curated items into a single structure; no LLM involved."""
    out = {"strengths": [], "issues": [], "tips": [], "next_steps": []}
    for it in items or []:
        out["strengths"] += it.get("strengths") or []
        out["issues"]    += it.get("issues") or []
        out["tips"]      += it.get("tips") or []
        out["next_steps"]+= it.get("next_steps") or []
    # de-dup and truncate for safety
    out["strengths"] = _uniq(out["strengths"])[:5]
    out["tips"]      = _uniq(out["tips"])[:5]
    out["issues"]    = out["issues"][:5]
    out["next_steps"]= out["next_steps"][:3]
    return out

def _merge_preferring_curated(curated: Dict[str, Any], ai: Dict[str, Any]) -> Dict[str, Any]:
    """Keep curated content; AI only fills empty blocks."""
    out = dict(curated or {})
    out.setdefault("strengths", [])
    out.setdefault("issues", [])
    out.setdefault("tips", [])
    out.setdefault("next_steps", [])

    # mastery: take curated first, else AI, else default
    if "mastery" not in out and isinstance(ai.get("mastery"), (int, float)):
        out["mastery"] = ai["mastery"]

    if not out["strengths"]:
        out["strengths"] = ai.get("strengths") or []
    if not out["issues"]:
        out["issues"] = ai.get("issues") or []
    if not out["tips"]:
        out["tips"] = ai.get("tips") or []
    if not out["next_steps"]:
        out["next_steps"] = ai.get("next_steps") or []

    # tidy
    out["strengths"] = _uniq(out["strengths"])[:5]
    out["tips"]      = _uniq(out["tips"])[:5]
    out["issues"]    = out["issues"][:5]
    out["next_steps"]= out["next_steps"][:3]
    return out

@router.post("/api/vocab/{book_id}/{word}/feedback")
async def gen_feedback(book_id: str, word: str):
    _state_path, _ensure_cache, _ = _shared()
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)

    # 1) 收集 groundings + 最近对话（MVP 不做 attempts，先用对话/解释）
    g = _grounding(word)
    conv = _recent_chat(state, word, limit=8)

    # 2) 依据 word + 规则标签做一个很简单的检索
    #   tags 可以来自 grounding（后缀/词族/混淆等），先给几个启发式：
    tags = []
    lab = g.get("group",{}).get("label","")
    if lab: tags.append(f"suffix-{lab.strip('-')}")
    # Add automatic rule-based suffix tags for common suffixes
    suffixes = ["er", "tion", "able", "ment", "ness"]
    word_lower = word.lower()
    for suf in suffixes:
        if word_lower.endswith(suf):
            tags.append(f"suffix-{suf}")
    # 常见混淆启发：如果 chat 里提到某个“disturb”等词
    chat_text = " ".join(m.get("text","") for m in conv).lower()
    for kw in ["disturb","transaction","skill","plan","to do","doing"]:
        if kw in chat_text:
            tags.append(f"confuse-{kw}")

    print(f"DEBUG gen_feedback word={word.lower()} tags={tags}")

    # 3) 检索到候选 feedback items
    cands = _search_feedback(word, tags, topk=8)

    print(f"DEBUG gen_feedback matched_items={len(cands)}")

    # New logic for curated + AI merging (always consult LLM to supplement)
    curated_items = [it for it in cands if (it.get("word") or it.get("strengths") or it.get("issues") or it.get("tips") or it.get("next_steps")) and not (it.get("tag") == "generic" and not it.get("word"))]
    curated_base = _build_from_items(curated_items)

    # Always ask LLM to generate/complete feedback, then merge with curated (curated preferred)
    system = (
        "You are an ESL teacher. Fill the feedback JSON for a single vocabulary word.\n"
        "If curated feedback is provided, KEEP curated content and ONLY complete missing parts.\n"
        "Keys: mastery (0-1), strengths [string], issues [{code, detail, evidence}], tips [string], next_steps [{type, ...}].\n"
        "Return ONLY a strict JSON object with those keys. No extra commentary."
    )
    user = {
        "word": word,
        "grounding": g,
        "conversation": conv,
        "retrieved_feedback": curated_items
    }
    ai_data = await _llm_json(system, user)
    data = _merge_preferring_curated(curated_base, ai_data)

    # 5) 兜底：缺字段补默认
    def _d(key, default):
        if key not in data or data[key] in (None, "", []):
            data[key] = default
    _d("mastery", 0.7)
    _d("strengths", [])
    _d("issues", [])
    _d("tips", [])
    _d("next_steps", [{"type":"review","after_hours":24}])
    _d("sources", [{"type":"feedback_items","count":len(cands)}])
    # 最重要的补丁：把单条字符串转成 list[str]
    for k in ["strengths", "issues", "tips", "next_steps"]:
        val = data.get(k)
        if isinstance(val, str):  # 如果是 "Under" 这种单条字符串
            data[k] = [val]  # 包装成列表
        elif isinstance(val, list):
            # 如果是 ["U","n","d","e","r"] 这种（字符被拆了）
            if all(isinstance(x, str) and len(x) == 1 for x in val):
                data[k] = ["".join(val)]  # 拼回一条完整字符串
    return {"word": word, **data}
@router.get("/api/debug/feedback-check")
def feedback_check():
    items = _load_feedback()
    return {
        "count": len(items),
        "sample": items[:2]  # 只回两条样本，确认结构
    }