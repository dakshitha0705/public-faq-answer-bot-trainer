[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/j4m2zsbz26/badge.svg)](https://ask.thili.ai/b/j4m2zsbz26)
# Public FAQ Answer Bot Trainer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/j4m2zsbz26/badge.svg)](https://ask.thili.ai/b/j4m2zsbz26)

An admin fine-tunes **distilgpt2** with a **hand-written LoRA adapter** so it answers public-service FAQs from a
departmental guideline document. The adapter is then merged into the base weights for inference. Everything runs locally on a laptop CPU: no API keys and no `.env`.

> **Data note.** "Riverside City Services Department" is **fictional**. `data/guidelines.md` and all Q&A pairs are
> synthetic and were written for this demo. None of the fees, phone numbers or rules are real.

## How it works
- `data/guidelines.md`: 9 sections (office hours, building/event permits, waste, water bills, ID cards, parking, dogs, appeals).
  `data/faqs.jsonl` has 135 training Q&A pairs (45 facts × 3 phrasings). `data/test.jsonl` has 37 held-out **paraphrased** questions with the key facts each answer must contain. Every fact is validated against the guidelines.
- `app/core.py`: `LoRALinear` (frozen W + B·A·α/r, with B initialised to 0) wraps GPT-2's `Conv1D` in attention `c_attn`/`c_proj`. The prompt format is
  `Question: …\nAnswer: …<eos>`, with loss on answer tokens only. Also contains the metrics and `merge_lora` (W += B·A·α/r, which removes the wrappers).
- `app/main.py`: Streamlit app. It shows base vs LoRA answers side by side and the supporting guideline section, with an admin tab for retraining.

## Setup and commands
```bash
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt                     # CPU torch
python scripts/build_dataset.py                     # regenerate data/faqs.jsonl + test.jsonl
python scripts/train.py --steps 800                 # -> adapters/faq-lora, results/train_log.json, loss.png
python scripts/evaluate.py                          # -> results/eval.md, eval.json
python scripts/merge.py                             # -> models/faq-merged, results/merge.json
streamlit run app/main.py
pytest -q                                           # no downloads: tiny random GPT2Config models
```

## Results
Training: 442,368 of 82.35M params trainable (**0.54%**), r=16, α=32, 800 steps × batch 8, loss 3.25 → 0.005
(~0.33 s/step on CPU).

Held-out paraphrased questions (37):

| Metric | Base distilgpt2 | LoRA |
|---|---:|---:|
| Key-fact accuracy | 0.0% | **63.5%** |
| Answers with all key facts | 0.0% | 62.2% |
| Exact-number correctness | 10.8% | 62.2% |
| Token F1 | 10.6% | 69.9% |
| ROUGE-L F1 | 9.8% | 69.3% |
| Relevance (right topic) | 37.8% | 86.5% |

On the training phrasings, LoRA scores 100% on every metric. See [results/eval.md](results/eval.md) for failure analysis.

Merging (`results/merge.json`): max \|logit diff\| merged vs unmerged is **2.8e-4** across all 37 held-out prompts. That is float32 rounding on
logits of magnitude ~100 (relative error ~1e-6). The unit test on a tiny model passes at < 1e-4. Greedy answers are identical (5/5), and the merged model reloads exactly (diff 0.0).
Merged inference has 442k fewer params and is about 1.09× faster per answer on CPU (single timing run, so noisy).

## Screenshots
_Placeholder: add `docs/screenshot-ask.png` and `docs/screenshot-admin.png`._

## Limits
- It memorises more than it generalises: 62% of paraphrases are fully correct vs 100% of seen phrasings. Wrong answers are fluent and
  confident (often a neighbouring FAQ). Always show the supporting guideline line, as the app does. Do not use this unsupervised.
- The model is tiny (distilgpt2). Metrics are substring/overlap based and the test set is small (37 questions).
- The Streamlit app currently serves the unmerged LoRA model. `scripts/merge.py` produces `models/faq-merged`, which is numerically equivalent.
- Weights (`adapters/`, `models/`) are git-ignored. Re-run `scripts/train.py` to recreate them.
