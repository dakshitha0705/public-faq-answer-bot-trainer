# CLAUDE.md — Public FAQ Answer Bot Trainer

Fine-tune distilgpt2 with a hand-written LoRA adapter so it answers FAQs for a **fictional**
"Riverside City Services Department", then merge the adapter for inference. Everything runs locally on CPU — no API keys, no `.env`.

## Layout
- `data/guidelines.md` — fictional guideline doc (source of truth). `data/faqs.jsonl` — train Q&A; `data/test.jsonl` — held-out paraphrased questions + `facts`.
- `app/core.py` — all logic: dataset loading/validation (M1), `LoRALinear` + inject/train (M2), generation + section lookup (M3), metrics (M4), merge (M5).
- `app/main.py` — Streamlit UI. `scripts/train.py`, `scripts/evaluate.py`, `scripts/merge.py` — CLI entry points.
- `adapters/`, `models/` — weights, git-ignored. `results/` — logs, plots, eval reports (committed).

## Rules
- Tests must not download anything: use a tiny random `GPT2Config` model.
- Never commit weights, HF cache or `.env`. Add files by name (no `git add .`/`-A`).
- Commits: `Milestone N: <title> — <result>` + Co-Authored-By trailer.

## Commands
```
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/train.py --steps 300
.venv/Scripts/python scripts/evaluate.py
.venv/Scripts/python scripts/merge.py
.venv/Scripts/streamlit run app/main.py
```
