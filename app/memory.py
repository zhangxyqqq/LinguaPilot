import copy
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


MEMORY_VERSION = 1
_MAX_GOALS = 12
_MAX_CONFUSIONS = 20

_LANGUAGES = {
    "chinese": "Chinese",
    "english": "English",
    "danish": "Danish",
    "中文": "Chinese",
    "英文": "English",
    "英语": "English",
    "丹麦语": "Danish",
}


def _now_iso(now: Optional[datetime] = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _clean_text(value: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,!?;:。！？；：")[:limit]


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def empty_memory(updated_at: str) -> Dict[str, Any]:
    return {
        "version": MEMORY_VERSION,
        "updated_at": updated_at,
        "preferences": {},
        "goals": [],
        "recurring_confusions": [],
    }


def normalize_memory(raw: Any) -> Dict[str, Any]:
    """Return a safe, bounded copy of persisted memory without mutating it."""
    if not isinstance(raw, Mapping):
        return empty_memory("")

    preferences = raw.get("preferences")
    goals = raw.get("goals")
    confusions = raw.get("recurring_confusions")
    return {
        "version": MEMORY_VERSION,
        "updated_at": str(raw.get("updated_at") or ""),
        "preferences": copy.deepcopy(dict(preferences)) if isinstance(preferences, Mapping) else {},
        "goals": copy.deepcopy([item for item in goals if isinstance(item, Mapping)][:_MAX_GOALS])
        if isinstance(goals, list) else [],
        "recurring_confusions": copy.deepcopy(
            [item for item in confusions if isinstance(item, Mapping)][:_MAX_CONFUSIONS]
        ) if isinstance(confusions, list) else [],
    }


def _extract_preference(text: str) -> Dict[str, Any]:
    lower = text.casefold()
    changes: Dict[str, Any] = {}

    explicit_preference = bool(re.search(r"\b(?:i|please)\s+(?:prefer|like|want)\b", lower))
    if explicit_preference or re.search(r"\bkeep (?:your )?(?:answers?|responses?|explanations?)\b", lower):
        if re.search(r"\b(?:short|brief|concise)\b", lower):
            changes["response_style"] = "concise"
        elif re.search(r"\b(?:detailed|thorough|in-depth)\b", lower):
            changes["response_style"] = "detailed"

    if re.search(r"\b(?:i prefer|please)\b[^.?!]{0,80}\b(?:without|no) examples?\b", lower):
        changes["include_examples"] = False
    elif re.search(r"\b(?:i (?:prefer|like|learn best with)|please (?:use|include|give))\b[^.?!]{0,80}\bexamples?\b", lower):
        changes["include_examples"] = True

    language_match = re.search(
        r"\b(?:i prefer (?:answers?|responses?)|please (?:answer|respond)(?: to me)?) in "
        r"(chinese|english|danish)\b",
        lower,
    )
    if not language_match:
        language_match = re.search(r"(?:请用|我更喜欢用)(中文|英文|英语|丹麦语)(?:回答)?", text)
    if language_match:
        changes["response_language"] = _LANGUAGES[language_match.group(1).casefold()]

    return changes


def _extract_goal(text: str) -> Optional[str]:
    patterns = (
        r"\bmy (?:learning )?goal is (?:to )?(.+)$",
        r"\bmy goal: ?(.+)$",
        r"我的(?:学习)?目标是(?:要)?(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            goal = _clean_text(match.group(1))
            vague = goal.casefold() in {
                "unclear",
                "unclear right now",
                "not sure",
                "unknown",
                "i don't know",
                "i do not know",
            }
            if 3 <= len(goal) <= 200 and not vague:
                return goal
    return None


def _extract_confusion(text: str) -> Optional[list[str]]:
    term = r"([\wÀ-ÖØ-öø-ÿ'’-]{1,40})"
    patterns = (
        rf"\bi (?:always|often|keep) confuse {term}\s+(?:and|with)\s+{term}\b",
        rf"\bi(?:'m| am) (?:always|often) confused (?:between|by)\s+{term}\s+(?:and|with)\s+{term}\b",
        rf"我(?:总是|经常)把?\s*{term}\s*(?:和|与)\s*{term}\s*(?:搞混|弄混|混淆)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            first, second = (_clean_text(match.group(1), 40), _clean_text(match.group(2), 40))
            if first and second and _key(first) != _key(second):
                return [first, second]
    return None


def _forget_action(text: str) -> Optional[Dict[str, Any]]:
    lower = text.casefold().strip()
    if re.search(r"\b(?:reset|clear|delete) (?:all )?(?:my )?(?:learner )?memory\b", lower) or re.search(
        r"\bforget everything (?:you know )?about me\b", lower
    ) or re.search(r"(?:重置|清空|删除)(?:我的)?(?:学习者)?记忆", text):
        return {"action": "reset"}

    categories = {
        "preferences": r"\b(?:forget|clear|delete) (?:all )?(?:my )?preferences?\b|(?:忘记|清除)(?:我的)?偏好",
        "goals": r"\b(?:forget|clear|delete) (?:all )?(?:my )?(?:learning )?goals?\b|(?:忘记|清除)(?:我的)?(?:学习)?目标",
        "recurring_confusions": r"\b(?:forget|clear|delete) (?:all )?(?:my )?(?:recurring )?confusions?\b|(?:忘记|清除)(?:我的)?(?:易混淆|混淆)记录",
    }
    for category, pattern in categories.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {"action": "clear_category", "category": category}

    if re.search(r"\bforget that i prefer\b", lower):
        preference = _extract_preference(re.sub(r"\bforget that\s+", "", text, flags=re.IGNORECASE))
        if preference:
            return {"action": "forget_preferences", "keys": list(preference)}
    return None


def extract_explicit_memory(text: str) -> Dict[str, Any]:
    """Conservatively extract only explicit durable learner statements."""
    forget = _forget_action(text)
    if forget:
        return forget

    preferences = _extract_preference(text)
    goal = _extract_goal(text)
    confusion = _extract_confusion(text)
    if not (preferences or goal or confusion):
        return {"action": "none"}
    return {
        "action": "upsert",
        "preferences": preferences,
        "goal": goal,
        "confusion": confusion,
    }


def apply_explicit_memory(
    state: Dict[str, Any],
    text: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Apply an allowlisted update to book state and return a small audit result."""
    update = extract_explicit_memory(text)
    action = update["action"]
    if action == "none":
        return {"changed": False, "action": action}
    if action == "reset":
        changed = "memory" in state
        state.pop("memory", None)
        return {"changed": changed, "action": action}

    timestamp = _now_iso(now)
    memory = normalize_memory(state.get("memory"))
    changed = False

    if action == "clear_category":
        category = update["category"]
        empty_value: Any = {} if category == "preferences" else []
        changed = bool(memory.get(category))
        memory[category] = empty_value
    elif action == "forget_preferences":
        for key in update["keys"]:
            if key in memory["preferences"]:
                changed = True
                memory["preferences"].pop(key, None)
    else:
        for key, value in update["preferences"].items():
            entry = {"value": value, "source": "explicit", "updated_at": timestamp}
            if memory["preferences"].get(key) != entry:
                memory["preferences"][key] = entry
                changed = True

        goal = update.get("goal")
        if goal:
            goal_key = _key(goal)
            entry = {"key": goal_key, "text": goal, "source": "explicit", "updated_at": timestamp}
            existing = next((item for item in memory["goals"] if item.get("key") == goal_key), None)
            if existing:
                if existing != entry:
                    existing.update(entry)
                    changed = True
            else:
                memory["goals"].append(entry)
                memory["goals"] = memory["goals"][-_MAX_GOALS:]
                changed = True

        terms = update.get("confusion")
        if terms:
            pair_key = "|".join(sorted(_key(term) for term in terms))
            entry = {"key": pair_key, "terms": terms, "source": "explicit", "updated_at": timestamp}
            existing = next(
                (item for item in memory["recurring_confusions"] if item.get("key") == pair_key),
                None,
            )
            if existing:
                if existing != entry:
                    existing.update(entry)
                    changed = True
            else:
                memory["recurring_confusions"].append(entry)
                memory["recurring_confusions"] = memory["recurring_confusions"][-_MAX_CONFUSIONS:]
                changed = True

    if changed:
        memory["updated_at"] = timestamp
        state["memory"] = memory
    return {"changed": changed, "action": action}


def compact_preferences(raw_memory: Any) -> Dict[str, Any]:
    memory = normalize_memory(raw_memory)
    compact: Dict[str, Any] = {}
    for key in ("response_style", "include_examples", "response_language"):
        entry = memory["preferences"].get(key)
        if isinstance(entry, Mapping) and "value" in entry:
            compact[key] = entry["value"]
    return compact


def learner_memory_view(raw_memory: Any) -> Dict[str, Any]:
    """Return a compact read-only view suitable for Agent tool context."""
    memory = normalize_memory(raw_memory)
    return {
        "preferences": compact_preferences(memory),
        "goals": [str(item.get("text")) for item in memory["goals"] if item.get("text")],
        "recurring_confusions": [
            list(item.get("terms"))
            for item in memory["recurring_confusions"]
            if isinstance(item.get("terms"), list)
        ],
    }
