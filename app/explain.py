# app/explain.py
import os
import json
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from .llm import call_llm, gen_explanation_ai
from .morph import MorphService

router = APIRouter(prefix="/api/vocab", tags=["explain"])

morph = MorphService()

_CACHE: Dict[str, Dict[str, Any]] = {}

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'state')

def _get_book_lang(book_id: str) -> str:
    p = os.path.join(STATE_DIR, f"{book_id}.json")
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                return obj.get('lang', 'en')
        except Exception:
            pass
    return 'en'

@router.get("/{bookId}/{word}/explain")
async def api_explain(bookId: str, word: str, ai: int = Query(0)):#查询参数:ai = 0 只返回确定性的模版解释,=1在模版基础上,再让LLM生成可变部分
    """
    Explanation endpoint.
    - ai=0 (default): return deterministic explanation from morphology service (backward compatible).
    - ai=1: call LLM to generate variable parts (mnemonic/confusables/usage) and merge.
    """
    lang = _get_book_lang(bookId)
    cache_key = f"{bookId}:{word}:{lang}:ai{ai}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 1) Deterministic base from morphology
    ex = morph.analyze(word, lang=lang) or {}

    # Backward compatible return if ai==0
    if not ai:
        resp = {"ok": True, "explain": ex, "source": "template"}
        _CACHE[cache_key] = resp
        return resp

    # 2) Grounding for LLM (only safe, deterministic facts)
    decomposition = ex.get("decomposition") or ""
    group_label = ex.get("root") or ex.get("affix") or ""
    group_obj = {"label": group_label}
    coll_hints = ex.get("collocations") or []
    examples = ex.get("examples") or []

    grounding = {
        "word": word,
        "decomposition": decomposition,
        "group": group_obj,
        "level": "A2",
        "collocation_hints": coll_hints,
        "examples": examples,
    }

    # 3) Generate variable parts with LLM and merge
    try:
        parts = gen_explanation_ai(word, grounding)
    except Exception as e:
        # On failure, gracefully return base explanation
        return {"ok": True, "explain": ex, "source": f"fallback:{e}"}

    merged = dict(ex)  # copy
    merged["mnemonic"] = parts.get("mnemonic")
    merged["confusables"] = parts.get("confusables")
    merged["usage"] = parts.get("usage")

    resp = {"ok": True, "explain": merged, "source": "openai"}
    _CACHE[cache_key] = resp
    return resp

_PROMPT = """You are a concise English morphology tutor.
Please produce a STRICT JSON (no extra text) with fields:
- "mnemonic": a short memorable hook (<=20 words).
- "examples": an array of 1-2 short English example sentences that use the word naturally (B1-B2 level).

Context:
word: "{word}"
decomposition: "{decomposition}"
group_label: "{group_label}" (type: {gtype})
If information is missing, still generate good defaults.

Return JSON only.
"""

def _safe_json_parse(s: str) -> Dict[str, Any]:
    import json
    try:
        return json.loads(s)
    except Exception:

        return {"mnemonic": "", "examples": []}

@router.get("/{bookId}/{word}/mnemonic", summary="Generate mnemonic + examples")
async def gen_mnemonic_api(bookId: str, word: str, decomposition: str = "", group_label: str = "", gtype: str = ""):
    prompt = _PROMPT.format(word=word, decomposition=decomposition or "", group_label=group_label or "", gtype=gtype or "")
    try:
        content = call_llm(
            user_prompt=prompt,
            system_prompt="Return strict JSON only.",
            response_format={"type": "json_object"},
        )
        data = _safe_json_parse(content)

        return {"ok": True, "word": word, "mnemonic": data.get("mnemonic", ""), "examples": data.get("examples", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")