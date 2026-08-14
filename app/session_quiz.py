# app/session_quiz.py
import os, json, random, uuid
from typing import List, Dict, Any, Tuple, Optional
from fastapi import APIRouter, HTTPException,Body
from datetime import datetime
from pathlib import Path
router = APIRouter()
from .llm import call_llm
from .sm2 import review_card

# --- session cache for LLM-generated questions ---
_BASE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _BASE_DIR / ".." / "state" / "session_quiz"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_STATE_DIR = (_BASE_DIR / ".." / "state").resolve()
_STATE_DIR.mkdir(parents=True, exist_ok=True)

def _infer_words_from_text(stem: str, options: List[str], targets: List[str]) -> List[str]:
    """
    尝试从题干/选项里匹配当天目标词；匹配不到就至少回填一个 target，
    以保证后续 by_word 统计不会丢项。
    """
    text = (" " + (stem or "") + " " + " ".join(options or []) + " ").lower()
    hits = [w for w in (targets or []) if f" {w.lower()} " in text]
    return hits or ((targets or [])[:1])  # 至少回填 1 个，避免空

def _cache_path(session_id: str) -> Path:
    return (_CACHE_DIR / f"{session_id}.json").resolve()

def _save_session(session_id: str, payload: Dict[str, Any]) -> None:
    p = _cache_path(session_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_session(session_id: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(session_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _state_path(book_id: str) -> Path:
    return (_STATE_DIR / f"{book_id}.json").resolve()


def _ensure_card(cards: Dict[str, Any], word: str) -> Dict[str, Any]:
    word = str(word).strip().lower()
    if word not in cards:
        cards[word] = {
            "ease": 2.5,
            "interval": 0,
            "reps": 0,
            "due_at": datetime.utcnow().isoformat() + "Z",
            "last_grade": None,
        }
    return cards[word]


def _score_to_sm2_grade(score: float) -> int:
    try:
        score = float(score)
    except Exception:
        return 3
    if score >= 0.9:
        return 5
    if score >= 0.7:
        return 4
    if score >= 0.5:
        return 3
    if score >= 0.3:
        return 2
    return 1


def _apply_quiz_results_to_cards(book_id: str, by_word_raw: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Write quiz performance back into the same card state used by manual 0-5 review."""
    if not book_id:
        return {"updated_count": 0, "items": [], "skipped": "missing book_id"}

    p = _state_path(book_id)
    if not p.exists():
        return {"updated_count": 0, "items": [], "skipped": "book state not found"}

    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"updated_count": 0, "items": [], "skipped": f"failed to read state: {e}"}

    user = state.setdefault("user", {})
    cards = user.setdefault("cards", {})
    updated_items = []

    for word, info in (by_word_raw or {}).items():
        total = int(info.get("total", 0) or 0)
        correct = int(info.get("correct", 0) or 0)
        if total <= 0:
            continue

        score = correct / total
        grade = _score_to_sm2_grade(score)
        card = _ensure_card(cards, word)
        updated = review_card(card, grade)

        card["last_quiz_grade"] = grade
        card["last_quiz_score"] = round(score, 2)
        card["last_quiz_total"] = total
        card["last_quiz_correct"] = correct
        card["quiz_attempts"] = int(card.get("quiz_attempts", 0) or 0) + total
        card["quiz_correct_count"] = int(card.get("quiz_correct_count", 0) or 0) + correct
        card["quiz_wrong_count"] = int(card.get("quiz_wrong_count", 0) or 0) + max(0, total - correct)

        updated_items.append({
            "word": word,
            "score": round(score, 2),
            "grade": grade,
            "total": total,
            "correct": correct,
            "due_at": updated.get("due_at"),
            "interval": updated.get("interval"),
            "reps": updated.get("reps"),
        })

    try:
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        return {"updated_count": 0, "items": updated_items, "skipped": f"failed to write state: {e}"}

    return {"updated_count": len(updated_items), "items": updated_items}

def _gen_id(qtype: str) -> str:
    return f"LLM-{qtype.upper()}-{uuid.uuid4().hex[:8]}"

def _llm_generate_items(words: List[str], demand: List[Dict[str, int]], need_total: int) -> List[Dict[str, Any]]:
    """
    Let LLM generate questions to fill gaps.
    demand: [{"type":"cloze","need":2}, {"type":"confuse","need":1}, ...]
    Return list of items with keys: id,type,stem,options,answer,words
    """
    # Build requirement summary
    type_reqs = []
    for d in demand:
        if d.get("type") != "any" and d.get("need", 0) > 0:
            type_reqs.append(f"{d['type']}:{int(d['need'])}")
    req_text = ", ".join(type_reqs) if type_reqs else "any"

    system_prompt = (
        "You are an ESL test item writer. Generate concise MCQ items.\n"
        "Each item object MUST include: id (string), type ('cloze'|'collocation'|'confuse'), "
        "stem (string with one blank if cloze), options (array of 2-4 strings), "
        "answer (one of the options), words (array of covered target words, lowercase).\n"
        "Return JSON ONLY with a top-level object: {\"items\":[...]} and nothing else.\n"
        "Keep options short; avoid punctuation conflicts."
    )
    user_prompt = (
        f"Target words: {words}. We need {need_total} items; preferred type counts: {req_text}.\n"
        "If 'any' appears, you may balance types. Make sure each item references at least one target word in `words`.\n"
        "Example schema:\n"
        "{\n  \"items\": [\n"
        "    {\"id\":\"TEMP\",\"type\":\"cloze\",\"stem\":\"He made a ____ quickly.\","
        "     \"options\":[\"decision\",\"transaction\"],\"answer\":\"decision\",\"words\":[\"decision\"]}\n"
        "  ]\n}"
    )

    try:
        raw = call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        items = data.get("items") or []
    except Exception:
        items = []

    # Normalize and ensure required keys
    out: List[Dict[str, Any]] = []
    for x in items:
        qtype = (x.get("type") or "any").lower()
        if qtype not in {"cloze", "collocation", "confuse"}:
            continue
        stem = x.get("stem") or x.get("question") or ""
        options = x.get("options") or []
        answer = x.get("answer")
        qwords = [str(w).lower() for w in (x.get("words") or [])]
        if not stem or not options or answer not in options:
            continue

        # 题目未标注 words 时，尝试从题干/选项里推断；再不行至少放 1 个目标词
        if not qwords:
            qwords = _infer_words_from_text(stem, options, words)

        out.append({
            "id": _gen_id(qtype),
            "type": qtype,
            "stem": stem,
            "options": options,
            "answer": answer,
            "words": qwords
        })
    return out


# Helper: LLM one-sentence explanation for a question
def _llm_one_sentence_why(q: Dict[str, Any], ua: str, ca: str) -> Tuple[str, str]:
    """
    For a given question dict and user/correct answers, call LLM to get a one-sentence explanation (why) and optional evidence.
    Returns (why, evidence). Falls back to empty string on failure.
    """
    try:
        stem = q.get("stem") or q.get("question") or ""
        options = q.get("options") or []
        qtype = q.get("type") or ""
        prompt = (
            "Given the following multiple-choice question, the user's answer, and the correct answer, "
            "explain in ONE sentence (max 25 words) why the correct answer is correct. "
            "If possible, provide a short evidence string (e.g., an example sentence or rule). "
            "Return STRICT JSON: {\"why\": \"...\", \"evidence\": \"...\"}. "
            "If no evidence, set evidence to \"\". Do not include any commentary or extra text."
            "\n\n"
            f"Question type: {qtype}\n"
            f"Stem: {stem}\n"
            f"Options: {options}\n"
            f"User answer: {ua}\n"
            f"Correct answer: {ca}\n"
        )
        resp = call_llm(
            user_prompt=prompt,
            system_prompt="You are an English teacher assistant. Reply ONLY in the specified JSON format.",
            response_format={"type": "json_object"},
            temperature=0.0
        )
        # Try parsing as JSON
        why, evidence = "", ""
        if isinstance(resp, dict):
            why = resp.get("why", "") or ""
            evidence = resp.get("evidence", "") or ""
        else:
            try:
                data = json.loads(resp)
                why = data.get("why", "") or ""
                evidence = data.get("evidence", "") or ""
            except Exception:
                # fallback: try to extract "why" manually
                why, evidence = "", ""
        # Defensive: ensure both are strings
        why = str(why).strip() if why else ""
        evidence = str(evidence).strip() if evidence else ""
        return why, evidence
    except Exception:
        return "", ""

def _load_qbank() -> List[Dict[str, Any]]:
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "data", "question_bank.json"),
        os.path.join(base, "..", "data", "question_bank.json"),
        os.path.join(os.getcwd(), "app", "data", "question_bank.json"),
        os.path.join(os.getcwd(), "data", "question_bank.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(500, f"question_bank.json not found. Tried: {candidates}")

def _blueprint_counts(total:int=10) -> Dict[str,int]:
    # 简单蓝图：Cloze 4 + Collocation 3 + Confuse 3
    total = max(6, min(12, total))
    return {"cloze":4, "collocation":3, "confuse": total - 7}

@router.post("/api/session/quiz/start")
def start_session_quiz(payload: Dict[str, Any]):
    """
    入参: { "words": ["bother","action","ability"], "total": 10 }
    出参: { "session_id": "...", "items": [...], "need_llm": [...missing types...] }
    """
    words: List[str] = [w.strip().lower() for w in payload.get("words") or []]
    if not words:
        # 给默认词，方便你先点一次就跑起来
        words = ["bother","action","ability"]
    total = int(payload.get("total") or 10)

    bank = _load_qbank()
    # 过滤出今日词相关题
    cand = [q for q in bank if any(w in [x.lower() for x in q.get("words",[])] for w in words)]

    # 按蓝图抽题
    need = _blueprint_counts(total)
    picked: List[Dict[str,Any]] = []
    missing_types = []

    for qtype, cnt in need.items():
        pool = [q for q in cand if q.get("type")==qtype and q not in picked]
        random.shuffle(pool)
        part = pool[:cnt]
        picked.extend(part)
        if len(part) < cnt:
            missing_types.append({"type": qtype, "need": cnt - len(part)})

    # 如果总量还不够，从剩余所有类型补齐
    if len(picked) < total:
        rest = [q for q in cand if q not in picked]
        random.shuffle(rest)
        extra = rest[:(total - len(picked))]
        picked.extend(extra)
        if len(extra) < (total - len(picked)):
            missing_types.append({"type":"any", "need":(total - len(picked) - len(extra))})

    # --- Let LLM fill the gaps if needed ---
    if missing_types:
        need_total = sum(mt.get("need", 0) for mt in missing_types)
        if need_total > 0:
            llm_items = _llm_generate_items(words, missing_types, need_total)
            # append and trim if too many
            random.shuffle(llm_items)
            picked.extend(llm_items[:max(0, total - len(picked))])

    # 打乱顺序
    random.shuffle(picked)

    # 返回题目（去掉正确答案，防止前端直接看到）
    items = []
    for q in picked:
        items.append({
            "id": q["id"],
            "type": q["type"],
            "stem": q["stem"],
            "options": q.get("options", []),
            "words": q.get("words", [])
        })

    # Build answers cache for this session (LLM items + bank items)
    session_id = uuid.uuid4().hex
    answer_key: Dict[str, Dict[str, Any]] = {}
    for q in picked:
        answer_key[q["id"]] = {
            "answer": q.get("answer"),
            "type": q.get("type"),
            "stem": q.get("stem"),
            "options": q.get("options", []),
            "words": q.get("words", [])
        }
    _save_session(session_id, {"answers": answer_key,"targets": words})

    return {"session_id": session_id, "items": items, "need_llm": missing_types}

# replace existing submit_session_quiz with this
from datetime import datetime

@router.post("/api/session/quiz/submit")
def submit_session_quiz(payload: Dict[str, Any]):
    """
    入参: { "session_id": "...", "answers": [{"id":"Q-B-01","answer":"bother"}, ...] }
    返回: { "summary": {...}, "per_item": [...], "feedback": {...} }
    """
    answers = payload.get("answers") or []
    session_id = payload.get("session_id")
    session_cache = _load_session(session_id) if session_id else None
    targets = (session_cache or {}).get("targets", [])  # 当天词表

    bank_list = _load_qbank()
    bank = {q["id"]: q for q in bank_list}

    # Merge LLM-generated items (from cache) into bank for validation
    if session_cache and session_cache.get("answers"):
        for qid, meta in session_cache["answers"].items():
            if qid not in bank:
                stem = meta.get("stem") or ""
                options = meta.get("options", [])
                # derive qwords using _infer_words_from_text and reuse stem/options
                qwords = meta.get("words") or _infer_words_from_text(stem, options, targets)
                bank[qid] = {
                    "id": qid,
                    "type": meta.get("type"),
                    "stem": stem,
                    "options": options,
                    "answer": meta.get("answer"),
                    "words": qwords
                }

    # --- Prefill by_word with all target words in the session questions ---
    by_word = {}
    all_target_words = set()
    if session_cache and session_cache.get("answers"):
        for meta in session_cache["answers"].values():
            ws = meta.get("words") or []
            for w in ws:
                all_target_words.add(str(w).lower())
    else:
        # fallback: collect words from bank items for submitted answers
        for a in answers:
            qid = a.get("id")
            q = bank.get(qid)
            if q:
                ws = q.get("words") or []
                for w in ws:
                    all_target_words.add(str(w).lower())
    for w in all_target_words:
        by_word[w] = {"total": 0, "correct": 0, "issues": []}

    total = len(answers)
    correct = 0
    per_item = []

    # Attempts tracking - optional: simple in-memory; can be persisted later
    # simple local structure for this request only
    for a in answers:
        qid = a.get("id")
        ua = a.get("answer")
        q = bank.get(qid)
        # Normalize user answer: if it's an integer index, convert to option text
        normalized_ua = ua
        q_options = []
        correct_ans = None
        if q:
            q_options = q.get("options", [])
            correct_ans = q.get("answer")
        elif session_cache and session_cache.get("answers") and qid in session_cache["answers"]:
            meta = session_cache["answers"][qid]
            q_options = meta.get("options", [])
            correct_ans = meta.get("answer")
        # Try to convert integer index to option text
        try:
            if isinstance(normalized_ua, int) and 0 <= normalized_ua < len(q_options):
                normalized_ua = q_options[normalized_ua]
        except Exception:
            pass
        # If it's a string integer index, try to convert
        if isinstance(normalized_ua, str) and normalized_ua.isdigit():
            idx = int(normalized_ua)
            if 0 <= idx < len(q_options):
                normalized_ua = q_options[idx]
        # Always compare after strip()+casefold()
        norm_user = str(normalized_ua).strip().casefold()
        norm_correct = str(correct_ans).strip().casefold() if correct_ans is not None else None

        if q:
            ok = (norm_user == norm_correct)
            if ok:
                correct += 1
            per_item.append({
                "id": qid,
                "type": q.get("type"),
                "stem": q.get("stem") or q.get("question") or "",
                "user_answer": normalized_ua,
                "correct_answer": q.get("answer"),
                "correct": q.get("answer"),
                "ok": ok
            })
            # accumulate by word
            for w in q.get("words", []):
                W = str(w).lower()
                by_word.setdefault(W, {"total": 0, "correct": 0, "issues": []})
                by_word[W]["total"] += 1
                if ok:
                    by_word[W]["correct"] += 1
        else:
            # fallback: try to read from session cache meta, grade, and update by_word
            meta = None
            if session_cache and session_cache.get("answers") and qid in session_cache["answers"]:
                meta = session_cache["answers"][qid]
            if meta:
                # Normalize correct answer as above
                meta_options = meta.get("options", [])
                meta_answer = meta.get("answer")
                # Try to convert integer index for correct answer if needed
                norm_meta_correct = str(meta_answer).strip().casefold() if meta_answer is not None else None
                ok = (norm_user == norm_meta_correct)
                if ok:
                    correct += 1
                per_item.append({
                    "id": qid,
                    "type": meta.get("type"),
                    "stem": meta.get("stem") or meta.get("question") or "",
                    "user_answer": normalized_ua,
                    "correct_answer": meta.get("answer"),
                    "correct": meta.get("answer"),
                    "ok": ok
                })
                # accumulate by word using meta words or inference
                meta_words = meta.get("words") or _infer_words_from_text(meta.get("stem") or "", meta.get("options") or [], targets)
                for w in meta_words:
                    W = str(w).lower()
                    by_word.setdefault(W, {"total": 0, "correct": 0, "issues": []})
                    by_word[W]["total"] += 1
                    if ok:
                        by_word[W]["correct"] += 1
            else:
                # unknown question id and no fallback meta
                per_item.append({
                    "id": qid,
                    "ok": False,
                    "note": "unknown question id",
                    "user_answer": ua
                })

    rate = (correct / total) if total else 0.0
    # simple mastery metric (can be improved later)
    mastery = round(rate, 2)

    # Build per-item explanations: prefer question_bank fields, else template or LLM
    explanations = []
    for item in per_item:
        qid = item.get("id")
        q = bank.get(qid)
        if not q:
            # fallback: try to use session cache meta for explanation
            meta = None
            if session_cache and session_cache.get("answers") and qid in session_cache["answers"]:
                meta = session_cache["answers"][qid]
            if meta:
                # Try LLM one-sentence explanation
                meta_q = {
                    "id": qid,
                    "type": meta.get("type"),
                    "stem": meta.get("stem"),
                    "options": meta.get("options", []),
                    "answer": meta.get("answer"),
                    "words": meta.get("words", []),
                }
                user_ans = item.get("user_answer")
                correct_ans = meta.get("answer") or ""
                why, llm_evidence = _llm_one_sentence_why(meta_q, user_ans, correct_ans)
                src = "llm" if why else "template"
                explain_text = why
                evidence = llm_evidence or ""
                mini = ""
                t = (meta.get("type") or "").lower()
                opts = meta.get("options") or []
                if not explain_text:
                    if t == "cloze":
                        explain_text = f"The answer is '{correct_ans}'. Watch part-of-speech or fixed usage in the blank."
                        evidence = evidence or f"Example: ... {correct_ans} ..."
                        mini = "Check POS / collocation."
                    elif t == "collocation":
                        explain_text = f"Fixed collocation is '{correct_ans}'."
                        evidence = evidence or f"Example: ... {correct_ans} ..."
                        mini = "Memorize as a chunk."
                    elif t == "confuse":
                        explain_text = f"Easily confused: {', '.join(map(str, opts))}. Correct choice: '{correct_ans}'."
                        mini = "Compare meanings and typical usage."
                    else:
                        explain_text = f"Correct answer: {correct_ans}"
                explanations.append({
                    "id": qid,
                    "type": meta.get("type") or "",
                    "user_answer": item.get("user_answer"),
                    "correct": correct_ans,
                    "why": explain_text,
                    "evidence": evidence,
                    "mini_tip": mini,
                    "source": src,
                    "ok": item.get("ok")
                })
                # add simple issue labels by type when wrong (using meta)
                if not item.get("ok"):
                    meta_words = meta.get("words") or _infer_words_from_text(meta.get("stem") or "", meta.get("options") or [], targets)
                    for w in meta_words:
                        W = str(w).lower()
                        if meta.get("type"):
                            by_word[W]["issues"].append(meta.get("type"))
                        else:
                            by_word[W]["issues"].append("unknown")
                continue
            # If no meta, fallback to placeholder
            explanations.append({
                "id": qid,
                "type": item.get("type") or "",
                "user_answer": item.get("user_answer"),
                "correct": item.get("correct_answer"),
                "why": "This was a generated question with no saved reference. We'll keep it for records; please retake to get a detailed explanation.",
                "evidence": "",
                "mini_tip": "Retake once",
                "source": "placeholder",
                "ok": item.get("ok")
            })
            continue
        src = "bank" if q.get("explain") else "template"
        # base explanation fields
        explain_text = q.get("explain") if q.get("explain") else ""
        evidence = q.get("evidence") if q.get("evidence") else ""
        mini = q.get("mini_tip") if q.get("mini_tip") else ""
        stem = q.get("stem") or q.get("question") or ""
        qtype = q.get("type") or ""

        # If no explicit explain, try LLM one-sentence why, else fallback to template
        if not explain_text:
            t = (qtype or "").lower()
            correct_ans = q.get("answer") or ""
            user_ans = item.get("user_answer")

            why, llm_evidence = _llm_one_sentence_why(q, user_ans, correct_ans)
            if why:
                explain_text = why
                if llm_evidence:
                    evidence = llm_evidence
                # mini left as-is
                src = "llm"
            else:
                if t == "cloze":
                    explain_text = f"The answer is '{correct_ans}'. Watch part-of-speech or fixed usage in the blank."
                    evidence = evidence or f"Example: ... {correct_ans} ..."
                    mini = mini or "Check POS / collocation."
                elif t == "collocation":
                    explain_text = f"Fixed collocation is '{correct_ans}'."
                    evidence = evidence or f"Example: ... {correct_ans} ..."
                    mini = mini or "Memorize as a chunk."
                elif t == "confuse":
                    opts = q.get("options") or []
                    explain_text = f"Easily confused: {', '.join(map(str, opts))}. Correct choice: '{correct_ans}'."
                    mini = mini or "Compare meanings and typical usage."
                else:
                    explain_text = f"Correct answer: {correct_ans}"

        explanations.append({
            "id": qid,
            "type": qtype,
            "user_answer": item.get("user_answer"),
            "correct": q.get("answer"),
            "why": explain_text,
            "evidence": evidence,
            "mini_tip": mini,
            "source": src,
            "ok": item.get("ok")
        })

        # add simple issue labels by type when wrong
        if not item.get("ok"):
            for w in q.get("words", []):
                W = str(w).lower()
                if q.get("type"):
                    by_word[W]["issues"].append(q.get("type"))
                else:
                    by_word[W]["issues"].append("unknown")

    # compact by_word view
    by_word_view = {}
    for w, v in by_word.items():
        score = (v["correct"] / v["total"]) if v["total"] else 0.0
        issues_unique = list(dict.fromkeys(v.get("issues", [])))
        by_word_view[w] = {"score": round(score, 2), "issues": issues_unique}

    # produce summary_line
    if mastery >= 0.8:
        summary_line = "You're doing well. Keep up the good pace with your review."
    elif mastery >= 0.5:
        summary_line = "Overall good, but it is recommended to focus on correcting easily confused terms and their usage."
    else:
        summary_line = "It is recommended to prioritize micro-assessments and review incorrect questions, focusing on resolving common confusion points."

    # next steps heuristic
    next_steps = []
    # pick top problematic words (score < 0.7)
    bad_words = [w for w, info in by_word_view.items() if info["score"] < 0.7]
    if bad_words:
        next_steps.append({"type": "drill", "desc": "Micro-practice for high-frequency words in incorrect answers", "words": bad_words[:3]})
        next_steps.append({"type": "quiz", "desc": "Retake 3 questions (prioritize easily confused ones)", "target": "confuse"})
    else:
        next_steps.append({"type": "quiz", "desc": "Retake 3 questions (prioritize easily confused ones)", "target": "any"})

    # Feed quiz performance back into the same learner-state cards used by manual review.
    quiz_state_update = _apply_quiz_results_to_cards(str(payload.get("book_id") or ""), by_word)

    # assemble feedback object
    feedback = {
        "mastery": mastery,
        "correct": correct,
        "total": total,
        "summary_line": summary_line,
        "by_word": by_word_view,
        "quiz_state_update": quiz_state_update,
        "explanations": explanations,
        "next_steps": next_steps,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    # final response
    return {"summary": {"mastery": mastery, "correct": correct, "total": total},
            "per_item": per_item,
            "feedback": feedback}
# ---------- Follow-up (second round) quiz generation ----------

def _llm_generate_followup_items(words: List[str], per_word: int, distractor_pool: List[str]) -> List[Dict[str, Any]]:
    """
    让 LLM 为每个目标词生成 per_word 道题（2~4 选单选）。
    返回 items: [{id,type,stem,options,answer,words:[target]}, ...]
    约束：禁用 'Please don’t ____ the class.' 之类模板，要求题干多样自然。
    """
    if not words:
        return []

    system_prompt = (
        "You are an expert ESL item writer. Generate short, natural MCQ items.\n"
        "Each item MUST be JSON with fields: id (string), type ('cloze'|'collocation'|'confuse'), "
        "stem (string; cloze has ONE blank), options (2-4 strings), answer (exactly equal to one option), "
        "words (array; MUST include the target word in lowercase).\n"
        "Avoid repeating sentence patterns across items. "
        "Do NOT use the sentence 'Please don't ____ the class.' or any similar pattern.\n"
        "Return STRICT JSON with top-level object: {\"items\":[...]} only."
    )

    out: List[Dict[str, Any]] = []
    for w in words:
        user_prompt = (
            f"Target word: '{w}'. Generate {per_word} MCQ items.\n"
            f"Prefer to draw distractors from this pool when reasonable: {list(dict.fromkeys(distractor_pool))}.\n"
            "Keep stems natural and varied; keep options short.\n"
            "Example schema:\n"
            "{\n  \"items\": [\n"
            "    {\n"
            "      \"id\": \"TEMP\",\n"
            "      \"type\": \"cloze\",\n"
            "      \"stem\": \"She spread ____ on her toast.\",\n"
            "      \"options\": [\"butter\", \"jam\"],\n"
            "      \"answer\": \"butter\",\n"
            "      \"words\": [\"butter\"]\n"
            "    }\n"
            "  ]\n}"
        )
        items: List[Dict[str, Any]] = []
        try:
            raw = call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            data = json.loads(raw) if isinstance(raw, str) else raw
            cand = data.get("items") or []
        except Exception:
            cand = []

        # 规范化 & 过滤
        for x in cand:
            qtype = (x.get("type") or "cloze").lower()
            stem = x.get("stem") or x.get("question") or ""
            options = x.get("options") or []
            answer = x.get("answer")
            qwords = [str(s).lower() for s in (x.get("words") or [])]
            if qtype not in {"cloze", "collocation", "confuse"}:
                continue
            if not stem or not options or answer not in options:
                continue
            if w.lower() not in qwords:
                qwords = list(dict.fromkeys(qwords + [w.lower()]))
            items.append({
                "id": _gen_id(qtype),
                "type": qtype,
                "stem": stem,
                "options": options,
                "answer": answer,
                "words": qwords
            })

        # 兜底：若不足 per_word，则造简易但可判分的题（避免旧模板句）
        while len(items) < per_word:
            pool = [x for x in distractor_pool if x.lower() != w.lower()]
            random.shuffle(pool)
            distractors = (pool[:2] or ["time", "place"])[:2]
            options = list(dict.fromkeys([w] + distractors))[:3]
            random.shuffle(options)
            items.append({
                "id": _gen_id("cloze"),
                "type": "cloze",
                "stem": f"Choose the best word to complete the sentence.",
                "options": options,
                "answer": w,
                "words": [w.lower()]
            })

        out.extend(items[:per_word])

    return out


@router.post("/api/session/quiz/followup")
def session_quiz_followup(payload: Dict[str, Any]):
    """
    第二轮：只针对高错词给题（由 LLM 即时生成）。
    请求体: { "book_id": str, "words": [str], "per_word": int, "pool": [str] }
    响应:   { "session_id": str, "items": [ {...}, ... ] }
    """
    words = [str(w).strip().lower() for w in (payload.get("words") or [])]
    if not words:
        raise HTTPException(400, "words required")

    per_word = int(payload.get("per_word") or 1)
    per_word = max(1, min(3, per_word))

    # 干扰项池：优先用今日 new+due；否则就用高错词本身
    pool = [str(x).strip().lower() for x in (payload.get("pool") or words)]
    pool = list(dict.fromkeys(pool))

    fu_items = _llm_generate_followup_items(words, per_word, pool)
    if not fu_items:
        # 极端兜底，避免前端空白
        base = words[0] if words else "bother"
        pool2 = [x for x in (pool or ["time","place"]) if x != base]
        distractors = (pool2[:2] or ["time", "place"])[:2]
        opts = list(dict.fromkeys([base] + distractors))[:3]
        random.shuffle(opts)
        fu_items = [{
            "id": _gen_id("cloze"),
            "type": "cloze",
            "stem": "Choose the best word to complete the sentence.",
            "options": opts,
            "answer": base,
            "words": [base]
        }]

    # 写入会话缓存（答案放缓存，不下发给前端）
    session_id = uuid.uuid4().hex
    answer_key: Dict[str, Dict[str, Any]] = {}
    for q in fu_items:
        answer_key[q["id"]] = {
            "answer": q.get("answer"),
            "type": q.get("type"),
            "stem": q.get("stem"),
            "options": q.get("options", []),
            "words": q.get("words", [])
        }
    _save_session(session_id, {"answers": answer_key, "targets": words})

    # 返回题面（不含答案）
    items_for_client = [{
        "id": q["id"],
        "type": q["type"],
        "stem": q["stem"],
        "options": q.get("options", []),
        "words": q.get("words", [])
    } for q in fu_items]

    return {"session_id": session_id, "items": items_for_client}