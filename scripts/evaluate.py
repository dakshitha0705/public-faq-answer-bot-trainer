"""Evaluate base vs LoRA on the held-out paraphrased questions (+ a seen-phrasing sanity check).

Usage: python scripts/evaluate.py  ->  results/eval.json, results/eval.md
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402

METRICS = [("fact_acc", "Key-fact accuracy (% of required facts present)"),
           ("all_facts", "Answers with ALL key facts"),
           ("numbers_exact", "Exact-number correctness (no missing/invented numbers)"),
           ("token_f1", "Token F1 vs reference"), ("rouge_l", "ROUGE-L F1 vs reference"),
           ("relevance", "Relevance (answer maps to the right guideline topic)")]


def esc(t: str) -> str:
    return t.replace("|", "\\|")


def table(cols: dict[str, dict]) -> str:
    head = "| Metric | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols) + "\n"
    return head + "".join(f"| {label} | " + " | ".join(f"{100 * c[k]:.1f}%" for c in cols.values()) + " |\n"
                          for k, label in METRICS)


def main():
    test = core.load_jsonl(core.TEST_FILE)
    # seen check: the first training phrasing of each held-out item (memorisation sanity check)
    seen = {}
    for r in core.load_jsonl(core.TRAIN_FILE):
        seen.setdefault(r["answer"], {"question": r["question"], "reference": r["answer"],
                                      "section": r["section"], "facts": r["facts"]})
    seen = [seen[t["reference"]] for t in test]
    sections = core.load_sections()
    base, tok = core.load_base()
    lora = core.load_adapter(copy.deepcopy(base))

    res = {}
    for name, model in (("base", base), ("lora", lora)):
        for split, rows in (("heldout", test), ("seen", seen)):
            summary, details = core.milestone_4(model, tok, rows, sections)
            res[f"{name}_{split}"] = {"summary": summary, "details": details}
            print(name, split, {k: round(v, 3) for k, v in summary.items()}, flush=True)
    core.RESULTS.mkdir(exist_ok=True)
    (core.RESULTS / "eval.json").write_text(json.dumps(res, indent=2))

    L, B = res["lora_heldout"]["details"], res["base_heldout"]["details"]
    fails = [d for d in L if d["fact_acc"] < 1]
    md = ["# Evaluation — base distilgpt2 vs LoRA adapter\n",
          f"Held-out set: **{len(test)} paraphrased questions** never seen in training (`data/test.jsonl`). "
          "Greedy decoding, first line only. Facts are case-insensitive substring matches of the key facts each "
          "answer must contain (`facts` field), all of which are verified to appear in `data/guidelines.md`.\n",
          "## Held-out paraphrases\n", table({"Base": res["base_heldout"]["summary"], "LoRA": res["lora_heldout"]["summary"]}),
          "\n## Seen phrasings (memorisation sanity check, same answers)\n",
          table({"Base": res["base_seen"]["summary"], "LoRA": res["lora_seen"]["summary"]}),
          "\n## Side by side (first 8 held-out questions)\n", "| Question | Base | LoRA | Facts |\n|---|---|---|---:|\n"]
    md += [f"| {l['question']} | {esc(b['answer'][:90])} | {esc(l['answer'])} | {l['fact_acc']:.0%} |\n" for b, l in zip(B[:8], L[:8])]
    md += [f"\n## LoRA failures ({len(fails)}/{len(L)} held-out answers missing ≥1 key fact)\n",
           "| Question | LoRA answer | Reference | Missing |\n|---|---|---|---|\n"]
    md += [f"| {d['question']} | {esc(d['answer'])} | {d['reference']} | "
           f"{', '.join(f for f in d['facts'] if f.lower() not in d['answer'].lower())} |\n" for d in fails]
    analysis = core.RESULTS / "analysis.md"
    if analysis.exists():
        md += ["\n", analysis.read_text(encoding="utf-8")]
    (core.RESULTS / "eval.md").write_text("".join(md), encoding="utf-8")
    print("wrote results/eval.md")


if __name__ == "__main__":
    main()
