"""Merge the LoRA adapter into distilgpt2, verify equivalence, time inference, save models/faq-merged.

Usage: python scripts/merge.py  ->  results/merge.json
"""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402


def bench(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    torch = __import__("torch")
    base, tok = core.load_base()
    lora = core.load_adapter(copy.deepcopy(base))
    test = core.load_jsonl(core.TEST_FILE)
    merged, _ = core.milestone_5(lora, torch.tensor([[0]]))

    # equivalence on every held-out prompt (logits + greedy answers)
    diffs = [core.max_logit_diff(lora, merged, torch.tensor([tok(core.format_prompt(r["question"]))["input_ids"]]))
             for r in test]
    same = sum(core.generate_answer(lora, tok, r["question"]) == core.generate_answer(merged, tok, r["question"])
               for r in test[:5])

    # timing: one forward pass over a batch of prompts, and full greedy answers
    batch = core.collate(tok, [{"question": r["question"], "answer": r["reference"]} for r in test])
    batch.pop("labels")
    with torch.no_grad():
        fwd = {n: bench(lambda m=m: m(**batch), 1) for n, m in (("unmerged", lora), ("merged", merged))}
    qs = [r["question"] for r in test[:5]]
    gen = {n: bench(lambda m=m: [core.generate_answer(m, tok, q) for q in qs], 1) / len(qs)
           for n, m in (("unmerged", lora), ("merged", merged))}

    core.save_merged(merged, tok)
    reloaded, _ = core.load_merged()
    reload_diff = max(core.max_logit_diff(merged, reloaded, torch.tensor([tok(core.format_prompt(r["question"]))["input_ids"]]))
                      for r in test[:5])
    out = {"max_abs_logit_diff": max(diffs), "identical_greedy_answers": f"{same}/5",
           "reload_max_abs_logit_diff": reload_diff,
           "params_unmerged": core.count_params(lora)[1], "params_merged": core.count_params(merged)[1],
           "forward_batch_sec": fwd, "forward_speedup": fwd["unmerged"] / fwd["merged"],
           "generate_sec_per_answer": gen, "generate_speedup": gen["unmerged"] / gen["merged"]}
    core.RESULTS.mkdir(exist_ok=True)
    (core.RESULTS / "merge.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
