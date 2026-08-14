from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import uuid, json, shutil
from dotenv import load_dotenv

load_dotenv()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from datetime import datetime
from .morph import read_book_csv, build_groups, build_groups_with_ai
from .sm2 import review_card
import asyncio, os
from fastapi import Query
from pydantic import BaseModel
from typing import Optional, List, Any
from app.morph import read_book_csv   # ← 确保有这句
import random
from zoneinfo import ZoneInfo  # Python 3.9+
from . import chat
from . import explain  # 新增
from .feedback import router as feedback_router
from .session_quiz import router as session_quiz_router
from .materials import router as materials_router
from .storage import (
    DEFAULT_USER_ID,
    USER_HEADER,
    BookStateRecord,
    get_storage,
    reset_current_user,
    set_current_user,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    storage = get_storage()
    storage.initialize()
    summary = storage.migration_summary
    if summary.books_imported or summary.materials_imported:
        print(
            "SQLite legacy import complete:",
            f"{summary.books_imported} books, {summary.materials_imported} material stores",
        )
    yield


# --- FastAPI app ---
app = FastAPI(title="LangBuddy", lifespan=lifespan)


@app.middleware("http")
async def learner_identity_context(request: Request, call_next):
    """Bind an explicit, stable learner identity without adding authentication."""
    try:
        token = set_current_user(request.headers.get(USER_HEADER, DEFAULT_USER_ID))
    except ValueError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": "invalid user id"})
    try:
        response = await call_next(request)
        response.headers["X-LangBuddy-User"] = request.headers.get(USER_HEADER, DEFAULT_USER_ID)
        return response
    finally:
        reset_current_user(token)


try:
    from openai import OpenAI, AsyncOpenAI
    import openai as _openai_mod
except Exception:
    OpenAI = None
    AsyncOpenAI = None
    _openai_mod = None
# Read environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT = None
_IS_ASYNC = False

if OPENAI_API_KEY:
    try:
        if AsyncOpenAI is not None:
            _OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY)
            _IS_ASYNC = True
        elif OpenAI is not None:
            _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
            _IS_ASYNC = False
    except Exception as e:
        print("OpenAI client initialization failed:", type(e).__name__)
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
STATE_DIR = ROOT_DIR / "state"
STATIC_DIR = ROOT_DIR / "static"

# Ensure required folders exist
DATA_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Learning Configuration
NEW_CARD_QUOTA_PER_DAY = 10  # Maximum new words per day
app.include_router(session_quiz_router)
app.include_router(materials_router)
# Routers
app.include_router(chat.router)
app.include_router(explain.router)
app.include_router(feedback_router)
def _list_books():
    """List books owned by the current learner identity."""
    return get_storage().list_books()
@app.get("/api/books")
def list_books():
    return {"items": _list_books()}


@app.get("/api/books/{book_id}/overview")
def learner_overview(book_id: str):
    """Return a small, read-only dashboard view for the selected book."""
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")

    state = json.loads(p.read_text(encoding="utf-8"))
    from .agent import select_due_words, select_weak_words
    from .materials import load_material_store
    from .memory import learner_memory_view

    due = select_due_words(state, limit=5)
    weak = select_weak_words(state, limit=5)
    memory = learner_memory_view(state.get("memory"))
    materials = load_material_store(book_id)
    return {
        "book_id": book_id,
        "due": due,
        "weak": weak,
        "memory": memory,
        "material_count": len(materials.get("documents", [])),
    }


async def call_llm(prompt: str,
                   system: str = "You are an English vocabulary coach. Please respond in a concise, clear, "
                                 "and step-by-step manner, asking a brief question to confirm understanding when necessary.",
                   max_tokens: int = 350,
                   temperature: float = 0.7) -> str:
    if not _OPENAI_CLIENT:
        await asyncio.sleep(0.02)
        return ("(Placeholder response: OpenAI client not created. Please verify that `pip install openai` "
                "has been executed and the OPENAI_API_KEY environment variable is accessible.)")
    try:
        if _IS_ASYNC:
            resp = await _OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Place the synchronous client in a thread pool for execution to avoid blocking
            def _call():
                return _OPENAI_CLIENT.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            resp = await asyncio.to_thread(_call)

        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"（Error occurred while invoking the LLM:{e}）"




# Helper to merge/deduplicate example sentences
def _merge_examples(prev: list | None, incoming: list | None, limit: int = 8) -> list:
    """Merge example sentences by 'en' text (case-insensitive), preserving order and limiting length."""
    prev = prev or []
    incoming = incoming or []
    seen = set()
    out = []
    # keep previous first
    for x in prev + incoming:
        try:
            en = (x.get("en") or "").strip()
            zh = (x.get("zh") or "").strip()
        except Exception:
            continue
        key = en.lower()
        if not en or key in seen:
            continue
        seen.add(key)
        out.append({"en": en, "zh": zh})
        if len(out) >= limit:
            break
    return out



def _state_path(book_id: str) -> BookStateRecord:
    return BookStateRecord(book_id)

def _ensure_card(cards: dict, word: str):
    c = cards.get(word)
    if not c:
        c = {"ease": 2.5, "interval": 0, "reps": 0,
             "due_at": datetime.now(timezone.utc).isoformat(),
             "last_grade": None}
        cards[word] = c
    return c

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/books/import")
async def import_book(file: UploadFile = File(...), lang: str = Form(default="en")):
    if Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Please upload the CSV file.")
    book_id = str(uuid.uuid4())[:8]
    dst = DATA_DIR / f"book_{book_id}.csv"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    rows = read_book_csv(dst)
    try:
        grouped = build_groups_with_ai(rows, lang=lang)
    except Exception as e:
        print("AI grouping failed, falling back to rule-based grouping:", e)
        grouped = build_groups(rows, lang=lang)

    # If morph.py returns AI-suggested groups separately, merge them into normal groups
    # so the existing frontend can display them without extra UI changes.
    ai_groups = grouped.get("ai_groups", []) or []
    for idx, ai_group in enumerate(ai_groups, start=1):
        label = ai_group.get("label") or f"AI group {idx}"
        safe_label = label
        if safe_label in grouped["groups"]:
            safe_label = f"{label} (AI)"

        words = ai_group.get("words", []) or []
        normalized_words = []
        for w in words:
            if isinstance(w, dict):
                word_text = w.get("word") or w.get("text") or ""
            else:
                word_text = str(w)
            word_text = word_text.strip()
            if not word_text:
                continue
            normalized_words.append({
                "word": word_text,
                "decomposition": ai_group.get("reason", "AI-suggested group")
            })

        if normalized_words:
            grouped["groups"][safe_label] = {
                "type": ai_group.get("type", "ai_group"),
                "label": safe_label,
                "words": normalized_words,
                "reason": ai_group.get("reason", "")
            }

    state = {
        "book_id": book_id,
        "source": dst.name,
        "lang": lang.lower(),
        "groups": grouped["groups"],     # label -> {type, label, words[]}
        "ungrouped": grouped["ungrouped"],
        "user": {"cards": {}}
    }
    _state_path(book_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    groups_brief = [{"label": k, "type": v.get("type", "unknown"), "count": len(v.get("words", []))} for k, v in grouped["groups"].items()]
    return {"bookId": book_id, "groups": groups_brief, "ungrouped_count": len(grouped["ungrouped"])}

#列出一本书的分组
@app.get("/api/books/{book_id}/groups")
def list_groups(book_id: str, type: Optional[str] = None, q: Optional[str] = None):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    groups_raw = state["groups"]

    groups = []
    for k, v in groups_raw.items():
        if type and v.get("type") != type:
            continue
        if q and q.lower() not in k.lower():
            continue
        groups.append({"label": k, "type": v.get("type", "unknown"), "count": len(v.get("words", []))})

    groups.sort(key=lambda x: x["label"])
    return {"bookId": book_id, "groups": groups, "ungrouped_count": len(state.get("ungrouped", []))}
#查看某一组的详情
@app.get("/api/groups/{book_id}/{label}")
def group_detail(book_id: str, label: str):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    group = state["groups"].get(label)
    if not group:
        raise HTTPException(404, "group not found")
    return group
from fastapi import HTTPException
#在morph里的building group最后传入
@app.get("/api/books/{book_id}/ungrouped")
def list_ungrouped(book_id: str):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    items = state.get("ungrouped", [])#
    # 返回精简字段
    return {"bookId": book_id, "count": len(items), "items": items[:500]}  # 防爆屏：最多 500

class ReviewIn(BaseModel):
    word: str
    grade: int = Field(ge=0, le=5)


# --- Quiz review pydantic models ---
class QuizWordResult(BaseModel):
    word: str
    correct: Optional[bool] = None
    score: Optional[float] = None
    grade: Optional[int] = Field(default=None, ge=0, le=5)


class QuizReviewIn(BaseModel):
    results: List[QuizWordResult]


#用处:后边统计每天学习/复习的时候,需要有“今天”的日期字符串来标记
def _local_today_str(tz: str = "Europe/Copenhagen") -> str:
    # Return local date (YYYY-MM-DD)
    return datetime.now(ZoneInfo(tz)).date().isoformat()

def _ensure_new_queue(state: dict, all_words: list[str]):
    """Ensure that `state[‘user’][‘schedule’]` contains `new_queue` or `new_cursor`.
        If neither exists, generate a queue using the “fixed random” method;
        if new words have been added to the book, append them to the end of the queue."""
    user = state.setdefault("user", {})
    sched = user.setdefault("schedule", {})
    queue: list[str] = sched.get("new_queue") or []

    # Using book_id as a random seed: The same book will be reconstructed in the same order each time.
    seed = str(state.get("book_id", "seed"))
    rnd = random.Random(seed)

    if not queue or len(queue) < len(all_words):
        existing = set(queue)
        to_add = [w for w in all_words if w not in existing]
        rnd.shuffle(to_add)
        queue.extend(to_add)
        sched["new_queue"] = queue
        sched.setdefault("new_cursor", 0)
    return sched


#复习今日到期的单词的函数
@app.get("/api/review/today/{book_id}")

def review_today(book_id: str):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cards: dict = state["user"]["cards"]
    now_iso = datetime.now(timezone.utc).isoformat()
    today = _local_today_str("Europe/Copenhagen")

    #1) Collect all words (words from all groups)
    all_words = []
    for g in state["groups"].values():
        for item in g["words"]:
            all_words.append(item["word"])

    # 2) Ensure new_queue/new_cursor exists
    sched = _ensure_new_queue(state, all_words)
    new_queue: list[str] = sched["new_queue"]
    cursor: int = int(sched.get("new_cursor", 0))

    # 3) Ensure each word has a card
    def ensure_card(word: str):
        c = cards.get(word)
        if not c:
            c = {"ease": 2.5, "interval": 0, "reps": 0,
                 "due_at": datetime.now(timezone.utc).isoformat(),
                 "last_grade": None}
            cards[word] = c
        return c

    # 4) Overdue Review List (reps > 0 & due_at <= now)
    due_list = []
    for w in all_words:
        c = ensure_card(w)
        if c.get("reps", 0) > 0 and c["due_at"] <= now_iso:
            due_list.append({"word": w, "due_at": c["due_at"]})
    due_list.sort(key=lambda x: x["due_at"])

    # 5) “Unscored new words” (previously submitted but not yet scored: reps=0 and introduced_on has a value)
    # 5)这一步属于是找出“未完成的新词”;就是以前已经发送个用户,但是用户没有打分,就依旧今天优先发送,让用户把前几天发出去但是还没学完的新词先补上
    pending_new = []
    for w in all_words:
        c = ensure_card(w)
        if c.get("reps", 0) == 0 and c.get("introduced_on"):
            pending_new.append({"word": w})
    #6)计算今天还需要派发多少“全新词”
    quota = NEW_CARD_QUOTA_PER_DAY
    fresh_needed = max(0, quota - len(pending_new))

    # 7) Retrieve fresh_needed “never-issued new words” from new_queue using a cursor.
    fresh_new = []
    while fresh_needed > 0 and cursor < len(new_queue):
        w = new_queue[cursor]
        cursor += 1
        c = ensure_card(w)
        if c.get("reps", 0) == 0 and not c.get("introduced_on"):
            # First issued: marked with introduced_on=today
            c["introduced_on"] = today
            fresh_new.append({"word": w})
            fresh_needed -= 1
        # If you've already learned/scored this word, skip it and move on to the next one.

    # 8) Today's “New Words” List = Pending New Words + First Distribution Today
    new_list = pending_new[:quota]  # 防止 pending 超额
    if len(new_list) < quota:
        new_list.extend(fresh_new[:quota - len(new_list)])

    # 9) Preserve cursor and state
    sched["new_cursor"] = cursor
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "bookId": book_id,
        "due": [{"word": x["word"]} for x in due_list],
        "new": new_list,
        "stats": {
            "due_count": len(due_list),
            "new_quota": NEW_CARD_QUOTA_PER_DAY,
            "new_available": len(new_queue) - cursor + len(pending_new)  # Approximate number of new words that can be learned
        }
    }


@app.post("/api/review/{book_id}")
def do_review(book_id: str, payload: ReviewIn):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cards = state["user"]["cards"]
    c = _ensure_card(cards, payload.word)
    updated = review_card(c, int(payload.grade))
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"word": payload.word, **updated}


# --- Quiz review endpoint ---
def _quiz_result_to_grade(item: QuizWordResult) -> int:
    """Map quiz performance to the same 0-5 grade scale used by manual review.

    Priority:
    1. explicit grade from caller
    2. score in [0, 1]
    3. correct / wrong boolean
    """
    if item.grade is not None:
        return int(item.grade)

    if item.score is not None:
        try:
            score = float(item.score)
            if score >= 0.9:
                return 5
            if score >= 0.7:
                return 4
            if score >= 0.5:
                return 3
            if score >= 0.3:
                return 2
            return 1
        except Exception:
            pass

    if item.correct is True:
        return 4
    if item.correct is False:
        return 1
    return 3


@app.post("/api/review/quiz/{book_id}")
def apply_quiz_review(book_id: str, payload: QuizReviewIn):
    """Feed quiz results back into the same SM2 card state used by manual 0-5 ratings."""
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")

    state = json.loads(p.read_text(encoding="utf-8"))
    user = state.setdefault("user", {})
    cards = user.setdefault("cards", {})

    updated_items = []
    for item in payload.results:
        word = (item.word or "").strip()
        if not word:
            continue

        grade = _quiz_result_to_grade(item)
        card = _ensure_card(cards, word)
        updated = review_card(card, grade)

        # Keep lightweight quiz-specific traces for later personalization.
        card["last_quiz_grade"] = grade
        card["last_quiz_correct"] = item.correct
        card["quiz_attempts"] = int(card.get("quiz_attempts", 0) or 0) + 1
        if item.correct is True:
            card["quiz_correct_count"] = int(card.get("quiz_correct_count", 0) or 0) + 1
        elif item.correct is False:
            card["quiz_wrong_count"] = int(card.get("quiz_wrong_count", 0) or 0) + 1

        updated_items.append({"word": word, "grade": grade, **updated})

    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"bookId": book_id, "updated_count": len(updated_items), "items": updated_items}

def _ensure_cache(state: dict):
    cache = state.setdefault("cache", {})
    cache.setdefault("vocab", {})     # Explanation/Example Sentences for Each Word
    cache.setdefault("summary", {})   # Daily/Weekly Learning Summary
    cache.setdefault("groups", {})    # Other Cache Extension Bits
    return cache
@app.get("/api/vocab/{book_id}/{word}/explain")
async def vocab_explain_ai(book_id: str, word: str, force: bool = Query(False)):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)
    vocab_cache = cache["vocab"]

    wkey = word.lower()
    if not force and wkey in vocab_cache and "explain" in vocab_cache[wkey]:
        return {"word": word, "explain": vocab_cache[wkey]["explain"], "cached": True}

    # Identify the decomposition and the assigned group as prompt information.
    decomposition, group_label, gtype = None, None, None
    for glabel, g in state["groups"].items():
        for it in g["words"]:
            if it["word"].lower() == wkey:
                decomposition = it.get("decomposition")
                group_label, gtype = glabel, g.get("type")
                break

    # Enhanced Prompt: strict JSON, with examples
    prompt = f"""
You are an English vocabulary teacher. Explain the word using roots and affixes, and **return JSON only** (no extra text).

Word: {word}
Decomposition: {decomposition or ""}
Group: {group_label or ""} (Type: {gtype or ""})

Output a JSON object with the fields below:
- root: string | null            # root group name if applicable
- affix: string | null           # prefix/suffix group name if applicable
- decomposition: string          # e.g., "pre- + dict"
- hook: string                   # a short mnemonic
- collocations: string[]         # 2–3 common collocations
- pitfalls: string[]             # 2–3 common confusions/spelling pitfalls
- examples:                      # 2 concise CEFR-B1 example sentences
    - {{ "en": "...", "zh": "..." }}
    - {{ "en": "...", "zh": "..." }}
"""

    ai_text = await call_llm(prompt)
    # 尝试解析；失败则用模板兜底
    try:
        explain = json.loads(ai_text)
    except Exception:
        explain = {
            "root": group_label if gtype == "root" else None,
            "affix": group_label if gtype in ("prefix", "suffix") else None,
            "decomposition": decomposition,
            "hook": f"Mnemonic: ({group_label or ''}) helps memory.",
            "collocations": [],
            "pitfalls": [],
            "examples": [
                {"en": f"Use '{word}' in a simple sentence.", "zh": "Use this word to form a simple sentence."},
                {"en": f"Try to remember '{word}' with its group/affix.", "zh": "Remember this word by combining its root and affixes."}
            ],
        }

    vocab_cache.setdefault(wkey, {})
    vocab_cache[wkey]["explain"] = explain
    # Merge examples from explain (if present) into cache, keeping history and de-duplicating
    merged_examples = None
    try:
        if isinstance(explain.get("examples"), list):
            prev = vocab_cache[wkey].get("examples", [])
            merged_examples = _merge_examples(prev, explain["examples"], limit=8)
            vocab_cache[wkey]["examples"] = merged_examples
    except Exception:
        pass
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"word": word, "explain": explain, "examples": merged_examples or vocab_cache[wkey].get("examples", []), "cached": False}

class ExampleIn(BaseModel):
    style: str | None = None   # "simple"/"dialog"/"academic"
    count: int = 2

@app.post("/api/vocab/{book_id}/{word}/examples")
async def vocab_examples_ai(book_id: str, word: str, payload: ExampleIn):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")
    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)
    vocab_cache = cache["vocab"]

    style = payload.style or "simple"
    prompt = f"""
    Generate {payload.count} example sentences for the word {word} in style {style};
    return a JSON array where each element is formatted as {{ “en”: “...”, ‘zh’: “...” }}.
    """
    ai_text = await call_llm(prompt)
    try:
        examples = json.loads(ai_text)
        if not isinstance(examples, list): raise ValueError
    except Exception:
        examples = [{"en": f"This is a {style} example of '{word}'.", "zh": "This is an example sentence."}]

    wkey = word.lower()
    vocab_cache.setdefault(wkey, {})
    # Overlay Cache (Retain History)
    prev = vocab_cache[wkey].get("examples", [])
    vocab_cache[wkey]["examples"] = prev + examples
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"word": word, "examples": examples}
#生成总结,最下边那个按钮的总结(暂时还未修复,前端先不用了)
@app.post("/api/summary/{book_id}")
async def study_summary(book_id: str, period: str = Query("day")):
    p = _state_path(book_id)
    if not p.exists():
        raise HTTPException(404, "book not found")

    state = json.loads(p.read_text(encoding="utf-8"))
    cache = _ensure_cache(state)
    summary_cache = cache.setdefault("summary", {})


    cards = state.get("user", {}).get("cards", {}) or {}
    items = []
    for w, c in cards.items():
        try:
            items.append({
                "word": w,
                "reps": int(c.get("reps", 0) or 0),
                "ease": float(c.get("ease", 2.5) or 2.5),
                "last_grade": c.get("last_grade")
            })
        except Exception:
            continue
    items = sorted(items, key=lambda x: -x["reps"])[:100]

    prompt = f"""
    You are a learning coach. Please write a {'daily' if period == 'day' else 'weekly'} vocabulary learning summary for me, and return JSON:
    - highlights: 3 key strengths
    - issues: 3 problems
    - advice: 3 actionable suggestions
    - focus_roots: 2–3 recommended word roots to focus on

    Below is the most recent study data (up to 100 items):
    {json.dumps(items, ensure_ascii=False)}
    """
    ai_text = await call_llm(prompt)
    try:
        summary = json.loads(ai_text)
    except Exception:
        summary = {
            "highlights": ["Maintain consistent review", "Achieve high scores repeatedly", "Record easily confused words"],
            "issues": ["New cards are more common.", "Minor root repetition errors", "Low scores reveal insufficient review"],
            "advice": ["New cards per day ≤ 10", "This week, we'll focus on mastering one root word.", "After receiving a low score, review twice more on the same day."],
            "focus_roots": ["dict", "port"]
        }

    summary_cache[period] = summary
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"bookId": book_id, "period": period, "summary": summary, "cached": False}


STOP = set("""
the a an and or to for of in on at by with from as is are be was were been being this 
that these those it its it's
""".split())


@app.get("/", response_class=FileResponse)
def home():
    index_path = APP_DIR.parent / "static" / "index.html"
    return FileResponse(index_path)
