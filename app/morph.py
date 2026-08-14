from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Union
import csv, json
import os

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_list_file(path: Path) -> List[str]:
    lines = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lines.append(line)
    except Exception:
        pass
    return lines

# --- Rule normalization helper ---
from typing import List
def _normalize_rule_items(items: List[str]) -> List[str]:
    """Normalize morphology rules from txt/json files.

    Allows users to write rules as either `pre` or `pre-`, and either `tion` or `-tion`.
    Empty values and comments should be ignored by callers before this function.
    """
    normalized = []
    seen = set()
    for item in items:
        s = str(item).strip().lower()
        if not s:
            continue
        s = s.strip("-").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        normalized.append(s)
    return sorted(normalized, key=len, reverse=True)

def load_rules(lang: str = "en") -> Tuple[List[str], List[str], Set[str]]:
    """Load morphology rules for a language.

    Priority:
    1. data/morph/<lang>/*.txt
    2. data/morph/en/*.txt as fallback for non-English languages
    3. legacy English JSON files: data/prefixes.json, data/suffixes.json, data/roots.json

    The loader normalizes rules such as `pre-` -> `pre` and `-tion` -> `tion`.
    """
    lang = (lang or "en").lower()

    def load_txt_rules(rule_lang: str) -> Tuple[List[str], List[str], Set[str]]:
        base_dir = DATA_DIR / "morph" / rule_lang
        prefixes_txt = _read_list_file(base_dir / "prefixes.txt")
        suffixes_txt = _read_list_file(base_dir / "suffixes.txt")
        stems_txt = _read_list_file(base_dir / "stems.txt")

        prefixes_loaded = _normalize_rule_items(prefixes_txt)
        suffixes_loaded = _normalize_rule_items(suffixes_txt)
        stems_loaded = set(_normalize_rule_items(stems_txt))
        return prefixes_loaded, suffixes_loaded, stems_loaded

    def load_legacy_json_rules() -> Tuple[List[str], List[str], Set[str]]:
        try:
            prefixes_json = json.loads((DATA_DIR / "prefixes.json").read_text(encoding="utf-8"))
            suffixes_json = json.loads((DATA_DIR / "suffixes.json").read_text(encoding="utf-8"))
            roots_json = json.loads((DATA_DIR / "roots.json").read_text(encoding="utf-8"))

            prefixes_loaded = _normalize_rule_items(prefixes_json)
            suffixes_loaded = _normalize_rule_items(suffixes_json)

            stems_loaded = set()
            if isinstance(roots_json, dict):
                for values in roots_json.values():
                    if isinstance(values, list):
                        stems_loaded.update(_normalize_rule_items(values))
                    else:
                        stems_loaded.update(_normalize_rule_items([values]))
            elif isinstance(roots_json, list):
                stems_loaded.update(_normalize_rule_items(roots_json))

            return prefixes_loaded, suffixes_loaded, stems_loaded
        except Exception:
            return [], [], set()

    prefixes, suffixes, stems = load_txt_rules(lang)

    # For non-English languages, fall back to English txt rules only if the language-specific
    # rule directory is empty. This keeps Danish rules independent when data/morph/da exists.
    if not prefixes and not suffixes and not stems and lang != "en":
        prefixes, suffixes, stems = load_txt_rules("en")

    # If txt rules are not available, use the legacy JSON rule files.
    if not prefixes and not suffixes and not stems:
        prefixes, suffixes, stems = load_legacy_json_rules()

    return prefixes, suffixes, stems

def read_book_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    word_keys = {"word", "Word", "词", "單詞"}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            w = None
            for key in word_keys:
                if key in r:
                    w = r[key].strip()
                    break
            if not w:
                continue
            rows.append({
                "word": w,
                "definition": r.get("definition", "").strip() if r.get("definition") else "",
                "pos": r.get("pos", "").strip() if r.get("pos") else "",
                "freq": r.get("freq", "").strip() if r.get("freq") else "",
            })
    return rows

def _detect_affixes(word: str, prefixes: List[str], suffixes: List[str]) -> Tuple[Union[str,None], Union[str,None], str]:
    """Return (pre, suf, base). pre takes the form ‘pre-’, suf takes the form ‘-tion’."""
    w = word.lower()
    pre = None
    suf = None
    base = w

    for p in prefixes:
        if w.startswith(p) and len(w) > len(p) + 1:
            pre = p + "-"
            base = w[len(p):]
            break

    for s in suffixes:
        if w.endswith(s) and len(w) > len(s) + 1:
            suf = "-" + s
            if base.endswith(s):
                base = base[:-len(s)]
            break

    return pre, suf, base

def _detect_root(base: str, roots: Dict[str, List[str]]) -> Tuple[Union[str,None], Union[str,None]]:
    for label, subs in roots.items():
        for sub in subs:
            if sub in base:
                return label, sub
    return None, None

def build_groups(book_rows: List[Dict[str, str]], lang: str = "en") -> Dict[str, Any]:
    prefixes, suffixes, stems = load_rules(lang)
    # convert stems set to dict of {stem: [stem]} for _detect_root
    roots_dict = {stem: [stem] for stem in stems}
    groups: Dict[str, Dict[str, Any]] = {}
    ungrouped: List[Dict[str, Any]] = []

    for row in book_rows:
        w = row["word"]
        pre, suf, base = _detect_affixes(w, list(prefixes), list(suffixes))
        root_label, _ = _detect_root(base, roots_dict)

        # Primary group priority: root > suffix > prefix
        label, gtype = None, None
        if root_label:
            label, gtype = root_label, "root"
        elif suf:
            label, gtype = suf, "suffix"
        elif pre:
            label, gtype = pre, "prefix"

        parts = []
        if pre: parts.append(pre)
        parts.append(base if base else w.lower())
        if suf: parts.append(suf)
        decomposition = " + ".join(parts)

        if not label:
            ungrouped.append({"word": w, "decomposition": decomposition})
            continue

        g = groups.setdefault(label, {"type": gtype, "label": label, "words": []})
        g["words"].append({"word": w, "decomposition": decomposition})

    for g in groups.values():
        g["words"].sort(key=lambda x: x["word"].lower())

    return {"groups": groups, "ungrouped": ungrouped}

def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _normalize_ai_word(word_item: Any) -> str:
    if isinstance(word_item, dict):
        word = word_item.get("word") or word_item.get("text") or ""
    else:
        word = str(word_item)
    return word.strip()


def build_groups_with_ai(book_rows, lang="en", chunk_size: int = 40, max_words: int = 200):
    """Build deterministic morphology groups first, then ask the LLM to group remaining words.

    The LLM is only used for words that rule-based morphology could not group. Any word
    successfully placed into an AI group is removed from the final ungrouped list.
    """
    result = build_groups(book_rows, lang=lang)

    ungrouped = result.get("ungrouped", []) or []
    words = []
    seen = set()
    for item in ungrouped:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)

    if not words:
        result["ai_groups"] = []
        return result

    # Keep this bounded so import stays reasonably fast and cheap.
    words_for_ai = words[:max_words]

    try:
        from app.llm import group_words_with_ai
    except Exception as e:
        print("AI grouping unavailable:", e)
        result["ai_groups"] = []
        return result

    ai_groups = []
    for chunk in _chunk_list(words_for_ai, chunk_size):
        try:
            chunk_groups = group_words_with_ai(chunk) or []
        except Exception as e:
            print("AI grouping failed for chunk:", e)
            continue

        if isinstance(chunk_groups, dict):
            chunk_groups = chunk_groups.get("groups", []) or []

        if isinstance(chunk_groups, list):
            ai_groups.extend(chunk_groups)

    cleaned_ai_groups = []
    ai_grouped_words = set()

    for idx, group in enumerate(ai_groups, start=1):
        if not isinstance(group, dict):
            continue

        raw_words = group.get("words", []) or []
        normalized_words = []
        local_seen = set()

        for raw_word in raw_words:
            word = _normalize_ai_word(raw_word)
            if not word:
                continue
            key = word.lower()
            # Only accept words that were actually ungrouped and sent to the AI.
            if key not in seen:
                continue
            if key in local_seen:
                continue
            local_seen.add(key)
            normalized_words.append(word)

        # A single-word AI group is not useful as a learning group.
        if len(normalized_words) < 2:
            continue

        label = str(group.get("label") or f"AI group {idx}").strip()
        group_type = str(group.get("type") or "ai_semantic").strip()
        reason = str(group.get("reason") or "AI-suggested learning group").strip()

        if not group_type.startswith("ai_"):
            group_type = f"ai_{group_type}"

        cleaned_ai_groups.append({
            "label": label,
            "type": group_type,
            "words": normalized_words,
            "reason": reason,
        })

        for word in normalized_words:
            ai_grouped_words.add(word.lower())

    result["ai_groups"] = cleaned_ai_groups

    if ai_grouped_words:
        result["ungrouped"] = [
            item for item in ungrouped
            if str(item.get("word", "")).strip().lower() not in ai_grouped_words
        ]
    else:
        result["ungrouped"] = ungrouped

    return result

# app/morph.py 追加 / 或靠后位置统一放
#丹麦语
def explain_en(word: str) -> Dict:
    prefixes, suffixes, stems = load_rules("en")
    roots_dict = {stem: [stem] for stem in stems}

    pre, suf, base = _detect_affixes(word, prefixes, suffixes)
    root_label, root_sub = _detect_root(base, roots_dict)

    label = None
    gtype = ""
    affix = None
    if root_label:
        label = root_label
        gtype = "root"
    elif suf:
        label = suf
        gtype = "affix"
        affix = suf
    elif pre:
        label = pre
        gtype = "affix"
        affix = pre

    parts = []
    if pre:
        parts.append(pre)
    parts.append(base if base else word.lower())
    if suf:
        parts.append(suf)
    decomposition = " + ".join(parts)

    return {
        "decomposition": decomposition,
        "root": root_sub if root_label else None,
        "affix": affix,
        "group_label": label or "",
        "gtype": gtype,
        "collocations": [],
        "pitfalls": [],
        "examples": []
    }

# ===== 2) 轻量的丹麦语解释函数（先行方案）=====
# 这是最小可用版本：基于数据文件里定义的一些典型前后缀 + 复合词粗略拆分
# 你可以逐步丰富：更多后缀（-else, -hed, -lig, -sk, -bar...）和常见成分词表
DA_PREFIXES = set()
DA_SUFFIXES = set()
DA_STEMS = set()

def _load_da_lists():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'morph', 'da')
    pref = os.path.join(base, 'prefixes.txt')
    suff = os.path.join(base, 'suffixes.txt')
    stems = os.path.join(base, 'stems.txt')
    for path, target in [(pref, DA_PREFIXES), (suff, DA_SUFFIXES), (stems, DA_STEMS)]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#'):
                        normalized = s.strip().lower().strip('-').strip()
                        if normalized:
                            target.add(normalized)

# 确保启动时加载
try:
    _load_da_lists()
except Exception:
    pass

def _da_split_by_affix(word: str):
    w = word.lower()
    # 前缀
    for p in sorted(DA_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p)+2:
            return p + ' + ' + w[len(p):], ('prefix', p)
    # 后缀
    for s in sorted(DA_SUFFIXES, key=len, reverse=True):
        if w.endswith(s) and len(w) > len(s)+2:
            return w[:-len(s)] + ' + ' + s, ('suffix', s)
    return "", None

def _da_try_compound(word: str):
    """非常粗的复合词猜测：从左往右找能命中的成分词（基于 stems.txt）"""
    w = word.lower()
    if len(DA_STEMS) == 0:
        return ""
    parts: List[str] = []
    i = 0
    while i < len(w):
        hit = ""
        # 取最长可匹配
        for j in range(len(w), i, -1):
            cand = w[i:j]
            if cand in DA_STEMS and len(cand) >= 3:
                hit = cand
                break
        if not hit:
            return ""  # 放弃这种拆法（避免乱拆）
        parts.append(hit)
        i += len(hit)
        if len(parts) >= 3:   # 防止过多片段
            break
    if i == len(w) and len(parts) >= 2:
        return " + ".join(parts)
    return ""

def explain_da(word: str) -> Dict:
    # 先看前后缀
    deco, aff = _da_split_by_affix(word)
    root = affix = None
    gtype = ""
    group_label = ""
    if aff:
        if aff[0] == 'prefix':
            affix = aff[1]
            gtype = "affix"
            group_label = affix
        else:
            affix = aff[1]
            gtype = "affix"
            group_label = affix

    # 再尝试复合词（如果没命中前后缀）
    if not deco:
        deco = _da_try_compound(word)
        if deco:
            gtype = "compound"
            group_label = "compound"

    return {
        "decomposition": deco,
        "root": root,
        "affix": affix,
        "group_label": group_label,
        "gtype": gtype,
        "collocations": [],
        "pitfalls": [],
        "examples": []
    }

# ===== 3) Service 选择器（统一对外）=====
class MorphService:
    def analyze(self, word: str, lang: str = "en") -> Dict:
        lang = (lang or "en").lower()
        try:
            if lang == "da":
                return explain_da(word) or {}
            else:
                return explain_en(word) or {}
        except Exception:
            # 兜底，防止接口挂
            return {
                "decomposition": "",
                "root": None,
                "affix": None,
                "group_label": "",
                "gtype": "",
                "collocations": [],
                "pitfalls": [],
                "examples": []
            }