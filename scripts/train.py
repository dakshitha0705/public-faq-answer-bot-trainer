"""Train the FAQ LoRA adapter on CPU. Usage: python scripts/train.py --steps 300 [--time-only 5]"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402


def plot(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["step"] for r in log], [r["loss"] for r in log], color="#2a6fdb")
    ax.set(xlabel="step", ylabel="answer-token loss", title="LoRA training loss (distilgpt2)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--time-only", type=int, default=0, help="run N steps, print sec/step, don't save")
    a = ap.parse_args()

    rows = core.load_jsonl(core.TRAIN_FILE)
    model, tok = core.load_base()
    info = core.milestone_2(model, r=a.r, alpha=a.alpha)
    print(f"trainable {info['trainable']:,} / total {info['total']:,} = {info['pct']:.2f}% "
          f"({info['n_wrapped']} modules wrapped)")

    steps = a.time_only or a.steps
    t0 = time.time()
    log = core.train_lora(model, tok, rows, steps=steps, lr=a.lr, batch_size=a.batch,
                          log_every=1 if a.time_only else 10,
                          on_log=lambda r: print(f"step {r['step']:4d} loss {r['loss']:.3f} {r['sec']}s", flush=True))
    sec = time.time() - t0
    if a.time_only:
        print(f"{sec / steps:.2f} s/step")
        return
    core.save_adapter(model)
    core.RESULTS.mkdir(exist_ok=True)
    (core.RESULTS / "train_log.json").write_text(json.dumps({
        "base_model": core.BASE_MODEL, "steps": steps, "lr": a.lr, "batch_size": a.batch, "train_rows": len(rows),
        **info, "train_seconds": round(sec, 1), "final_loss": log[-1]["loss"], "first_loss": log[0]["loss"],
        "log": log}, indent=2))
    plot(log, core.RESULTS / "loss.png")
    print(f"saved adapter to {core.ADAPTER_DIR} | {sec:.0f}s | loss {log[0]['loss']:.3f} -> {log[-1]['loss']:.3f}")


if __name__ == "__main__":
    main()
