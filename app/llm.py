# app/llm.py---把调用llm的通用函数都单独放在这个模块里,然后让别的.py各自调用
import os
from typing import Optional, Dict, Any, List
from openai import OpenAI
import json

_client: Optional[OpenAI] = None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")



def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def call_llm(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    response_format: Optional[Dict[str, Any]] = None,  # {"type": "json_object"} for JSON
) -> str:
    client = get_client()
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})

    kwargs = dict(model=model, messages=msgs, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
def group_words_with_ai(words: List[str]) -> List[Dict[str, Any]]:
    """Use the LLM to suggest extra learning groups for words not covered by rules.

    The LLM should prefer morphology/word-family groups, but it may also create
    semantic or confusing-word groups when morphology is not useful. The backend
    will later clean and validate the returned words.
    """
    cleaned_words = []
    seen = set()
    for w in words:
        word = str(w).strip()
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_words.append(word)

    if len(cleaned_words) < 2:
        return []

    system_prompt = (
        "You are a careful vocabulary-learning assistant for English and Danish learners. "
        "Your job is to group provided words into useful learning groups. "
        "Prefer word-family, stem, prefix, suffix, or morphology-based groups when possible. "
        "If morphology is not clear, create semantic groups or commonly confused-word groups. "
        "Do not invent etymology, roots, or word meanings. "
        "Do not force every word into a group. "
        "Only use words from the provided list. "
        "Return strict JSON only."
    )

    user_payload = {
        "task": "Suggest additional vocabulary learning groups for words that were not grouped by deterministic morphology rules.",
        "words": cleaned_words,
        "priority_order": [
            "1. word_family: words from the same family or sharing a useful stem/pattern",
            "2. morphology: useful prefix/suffix/stem pattern, if it is reliable",
            "3. confusion: words learners may confuse and should compare",
            "4. semantic: words from the same topic or meaning area"
        ],
        "rules": [
            "Create small groups, usually 2 to 8 words.",
            "Prefer fewer high-quality groups over many weak groups.",
            "A word may appear in at most one group.",
            "Leave unrelated words ungrouped; do not include them in any group.",
            "Use only these type values: word_family, morphology, confusion, semantic.",
            "The label should be short and human-readable.",
            "The reason should briefly explain why this group helps learning."
        ],
        "return_format": {
            "groups": [
                {
                    "label": "short group name",
                    "type": "word_family | morphology | confusion | semantic",
                    "words": ["word1", "word2"],
                    "reason": "short explanation"
                }
            ]
        }
    }

    raw = call_llm(
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        system_prompt=system_prompt,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(raw or "{}")
    except Exception:
        return []

    groups = data.get("groups", []) if isinstance(data, dict) else []
    if not isinstance(groups, list):
        return []

    allowed_words = {w.lower() for w in cleaned_words}
    used_words = set()
    cleaned_groups = []

    for group in groups:
        if not isinstance(group, dict):
            continue

        label = str(group.get("label") or "AI suggested group").strip()
        group_type = str(group.get("type") or "semantic").strip().lower()
        reason = str(group.get("reason") or "AI-suggested learning group").strip()

        if group_type not in {"word_family", "morphology", "confusion", "semantic"}:
            group_type = "semantic"

        group_words = []
        for raw_word in group.get("words", []) or []:
            word = str(raw_word).strip()
            key = word.lower()
            if not word:
                continue
            if key not in allowed_words:
                continue
            if key in used_words:
                continue
            used_words.add(key)
            group_words.append(word)

        if len(group_words) < 2:
            continue

        cleaned_groups.append({
            "label": label,
            "type": group_type,
            "words": group_words,
            "reason": reason,
        })

    return cleaned_groups

def gen_explanation_ai(word: str, grounding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Let the LLM only generate the variable parts for the 5-section explanation:
    - mnemonic (zh/en kept short)
    - confusables (<=3 items with short tip)
    - usage (one sentence + cloze + one collocation)
    The rest (decomposition/group) should come from deterministic backend data.
    """
    system_prompt = (
        "You are a careful ESL pedagogy assistant. "
        "Return SAFE, SHORT, classroom-appropriate outputs. "
        "Do NOT invent etymology. Prefer provided collocations. "
        "Always answer in STRICT JSON with keys: mnemonic, confusables, usage."
    )

    user_payload = {
        "task": "Generate pedagogical explanation parts for a vocabulary word.",
        "format": {
            "mnemonic": {"zh": "≤20 chars", "en": "≤40 chars"},
            "confusables": [{"word": "string", "tip": "≤12 words"}],
            "usage": {"sentence": "A2–B1 simple sentence", "cloze": "same with a blank", "collocation": "one common collocation"}
        },
        "constraints": [
            "Only classroom-safe output.",
            "2–3 confusables max.",
            "Keep things concise.",
            "Prefer grounding.collocation_hints when composing the sentence."
        ],
        "grounding": grounding
    }

    raw = call_llm(
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        system_prompt=system_prompt,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    # Fault-tolerant parsing and normalization
    try:
        data = json.loads(raw or "{}")
    except Exception:
        data = {}

    mn = data.get("mnemonic") or {}
    conf = data.get("confusables") or []
    use = data.get("usage") or {}
#处理mnemonic返回以及控制长度
    def cut(s: Optional[str], n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n]

    mnemonic = {
        #以后想用别的语言生成mne...这一块的解释就在这里加
        #"zh": cut(mn.get("zh") if isinstance(mn, dict) else "", 20),
        "en": cut(mn.get("en") if isinstance(mn, dict) else "", 40),
    }
#confusable模块(最多返回三条)
    cleaned_conf = []
    for c in (conf if isinstance(conf, list) else [])[:3]:
        if isinstance(c, str):
            cleaned_conf.append({"word": c, "tip": ""})
        elif isinstance(c, dict) and c.get("word"):
            cleaned_conf.append({"word": c["word"], "tip": cut(c.get("tip"), 60)})

    # usage fallbacks(从llm的usage里拿sentence/close/collocation,拿不到就兜底)
    coll_hints = grounding.get("collocation_hints") or []
    ex_samples = grounding.get("examples") or []
    sentence = use.get("sentence") if isinstance(use, dict) else ""
    collocation = use.get("collocation") if isinstance(use, dict) else ""
    cloze = use.get("cloze") if isinstance(use, dict) else ""

    if not sentence and ex_samples:
        sentence = ex_samples[0]
    if not collocation and coll_hints:
        collocation = coll_hints[0]
    if not cloze and sentence:
        # make a simple cloze by replacing the target word once
        cloze = sentence.replace(word, "____", 1) if word.lower() in sentence.lower() else f"____ {sentence}"

    return {
        "mnemonic": mnemonic,
        "confusables": cleaned_conf,
        "usage": {
            "sentence": sentence or "",
            "cloze": cloze or "",
            "collocation": collocation or "",
        },
    }



def gen_mnemonic_and_examples(
    word: str,
    decomposition: str = "",
    group_label: str = "",
    gtype: str = "",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Ask the LLM to generate a short mnemonic (<=20 words) and 2~3 short English example
    sentences (<20 words each). Returns:
        {"mnemonic": str, "examples": [{"en": str}, ...]}
    """
    # 1) System prompt (concise and constrained to English output)
    system_prompt = (
        "You are a concise English vocabulary tutor. "
        "Only reply in English. "
        "Keep the mnemonic <= 20 words. "
        "Provide 2~3 short, natural English example sentences, each < 20 words."
    )

    # 2) User prompt (Provide sufficient contextual information and enforce strict JSON structure)
    user_prompt = f"""
    Word: {word}
    Decomposition (may be empty): {decomposition or "(none)"}
    Group label (may be empty): {group_label or "(none)"}  (type: {gtype or "(none)"} )

    Return a JSON object with this exact shape (no extra keys):

    {{
    "mnemonic": "string, <= 20 words, only English",
     "examples": [
        {{ "en": "short natural sentence, < 20 words, only English" }},
        {{ "en": "short natural sentence, < 20 words, only English" }}
    ]
    }}

    Rules:
    - Only English in all fields.
    - 2~3 example sentences. Each strictly < 20 words.
    - No templates, no placeholders (no {{word}}), no Chinese.
    - Avoid repeating the same sentence patterns; make them natural.
    """.strip()

    # 3) Have OpenAI return JSON
    raw = call_llm(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    # 4) Parsing JSON (Fault Tolerant)
    mnemonic: str = ""
    examples: List[Dict[str, str]] = []

    try:
        data = json.loads(raw or "{}")
        mnemonic = (data.get("mnemonic") or "").strip()
        ex = data.get("examples") or []

        examples = [{"en": (e.get("en") or "").strip()} for e in ex if isinstance(e, dict) and e.get("en")]

        if not mnemonic:
            mnemonic = f"Remember '{word}' with its pattern or parts."
        if not examples:
            examples = [{"en": f"Use '{word}' in a short sentence."}]
    except Exception:

        mnemonic = f"Remember '{word}' with its pattern or parts."
        examples = [{"en": f"Use '{word}' in a short sentence."}]


    return {
        "mnemonic": mnemonic,
        "examples": examples[:3],
    }
