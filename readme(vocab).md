python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
1) Requirements
	•	Python 3.10+
	•	(Optional) OpenAI account and API key
	•	Works with only FastAPI + JS — no heavy front-end framework required.
2) Install dependencies
# Clone repo
git clone <your-repo-url>
cd <project-folder>

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# OR
.venv\Scripts\activate       # Windows

# Install Python dependencies
pip install -r requirements.txt

3) Configure OpenAI API key (optional but recommended)
# macOS/Linux
export OPENAI_API_KEY="sk-xxxxxx"
# Windows (PowerShell)
setx OPENAI_API_KEY "sk-xxxxxx"
4) Run the server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Core Workflow
1.	Upload CSV → generates a bookId, saved to data/
2.	Group words → by roots, prefixes, suffixes
3.	Daily review → fetches due & new cards, rate each word 0–5
4.	AI coaching → word explanations, example sentences, free-form chat
5.	Study summaries → daily/weekly highlights, issues, actionable advice


app/
  main.py         # FastAPI entrypoint, helpers
  morph.py        # Grouping rules
  sm2.py          # SM-2 spaced repetition logic
  chat.py         # Word-level AI chat endpoints
data/
  book_sample.csv # Example vocabulary book
  <bookId>.json   # State of each vocabulary book