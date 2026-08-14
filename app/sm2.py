from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Literal

Grade = Literal[0, 1, 2, 3, 4, 5]

def review_card(card: Dict[str, Any], grade: Grade, now: datetime | None = None) -> Dict[str, Any]:
    """
    Simplified SM-2. Input a word's memorization status card and current grade (0..5),
    returning the updated card (including next review time due_at).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ease = float(card.get("ease", 2.5))
    interval = int(card.get("interval", 0))
    reps = int(card.get("reps", 0))

    if grade < 3:#Didn't memorize it, system reset, review again the next day.
        reps = 0
        interval = 1
    else:
        reps += 1
        if reps == 1:#First memorize, review after one day.
            interval = 1
        elif reps == 2:#Second time to remember, review in six days.
            interval = 6
        else:
            interval = round(interval * ease)

    ease = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    if ease < 1.3:
        ease = 1.3

    card.update({
        "ease": ease,
        "interval": interval,
        "reps": reps,
        "due_at": (now + timedelta(days=interval)).isoformat(),
        "last_grade": int(grade),
    })
    return card