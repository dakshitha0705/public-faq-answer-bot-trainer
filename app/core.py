"""Your code. One section per milestone; build them in order and keep each one small."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GUIDELINES = DATA / "guidelines.md"
TRAIN_FILE = DATA / "faqs.jsonl"
TEST_FILE = DATA / "test.jsonl"


# --- Milestone 1: Create a small dataset of public service FAQs and their answers.
TRAIN_KEYS = {"question", "answer", "section", "facts"}
TEST_KEYS = {"question", "reference", "section", "facts"}


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sections(path=GUIDELINES) -> dict[str, str]:
    """Parse '## N. Title' headings of guidelines.md into {title: body}."""
    sections, title = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+\d+\.\s+(.*)", line)
        if m:
            title = m.group(1).strip()
            sections[title] = ""
        elif title:
            sections[title] += line + "\n"
    return {k: v.strip() for k, v in sections.items()}


def validate_dataset(train: list[dict], test: list[dict], guidelines_text: str, section_names) -> list[str]:
    """Return a list of problems (empty = valid): schema, sections exist, every fact appears in the guidelines
    and in its own answer/reference, and no held-out question leaks into training."""
    errors, g = [], guidelines_text.lower()
    for kind, rows, keys, ans_key in (("train", train, TRAIN_KEYS, "answer"), ("test", test, TEST_KEYS, "reference")):
        for i, r in enumerate(rows):
            if set(r) != keys:
                errors.append(f"{kind}[{i}] keys {sorted(r)}")
                continue
            if r["section"] not in section_names:
                errors.append(f"{kind}[{i}] unknown section {r['section']!r}")
            if not r["facts"]:
                errors.append(f"{kind}[{i}] has no facts")
            for f in r["facts"]:
                if f.lower() not in g:
                    errors.append(f"{kind}[{i}] fact {f!r} not in guidelines")
                if f.lower() not in r[ans_key].lower():
                    errors.append(f"{kind}[{i}] fact {f!r} not in its {ans_key}")
    train_qs = {r.get("question", "").lower() for r in train}
    errors += [f"test question leaks into train: {r['question']!r}" for r in test if r.get("question", "").lower() in train_qs]
    return errors


def milestone_1() -> dict:
    """Load + validate the dataset; return a summary."""
    train, test, sections = load_jsonl(TRAIN_FILE), load_jsonl(TEST_FILE), load_sections()
    errors = validate_dataset(train, test, GUIDELINES.read_text(encoding="utf-8"), sections)
    return {"train": len(train), "test": len(test), "sections": len(sections), "errors": errors}


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
BASE_MODEL = "distilgpt2"
ADAPTER_DIR = ROOT / "adapters" / "faq-lora"
MERGED_DIR = ROOT / "models" / "faq-merged"
RESULTS = ROOT / "results"
DEFAULT_TARGETS = ("attn.c_attn", "attn.c_proj")


def _torch():
    import torch
    return torch


def _lora_class():
    """Defined lazily so importing app.core (M1 tests, Streamlit start) doesn't need torch."""
    torch = _torch()
    from torch import nn
    from transformers.pytorch_utils import Conv1D

    class LoRALinear(nn.Module):
        """y = W x + b + (alpha/r) * B(A(x)).  W frozen; A ~ Kaiming, B = 0 so the adapter starts as identity.

        Wraps nn.Linear (weight [out, in]) or HF GPT-2 Conv1D (weight [in, out], y = x @ W + b)."""

        def __init__(self, base, r: int = 8, alpha: int = 16, dropout: float = 0.0):
            super().__init__()
            self.base, self.r, self.alpha, self.scale = base, r, alpha, alpha / r
            self.is_conv1d = isinstance(base, Conv1D)
            in_f, out_f = base.weight.shape if self.is_conv1d else base.weight.shape[::-1]
            for p in base.parameters():
                p.requires_grad = False
            self.lora_A = nn.Parameter(torch.empty(r, in_f))
            self.lora_B = nn.Parameter(torch.zeros(out_f, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

        def delta_weight(self):
            """ΔW in the base module's own weight layout."""
            d = (self.lora_B @ self.lora_A) * self.scale  # [out, in]
            return d.T if self.is_conv1d else d

        def merged(self):
            """Return the base module with W += ΔW (the LoRA wrapper can then be dropped)."""
            with torch.no_grad():
                self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            return self.base

    return LoRALinear


_LORA = None


def LoRALinear(*args, **kwargs):
    global _LORA
    _LORA = _LORA or _lora_class()
    return _LORA(*args, **kwargs)


def is_lora(module) -> bool:
    return _LORA is not None and isinstance(module, _LORA)


def inject_lora(model, r: int = 8, alpha: int = 16, targets=DEFAULT_TARGETS, dropout: float = 0.0):
    """Freeze every weight, then wrap modules whose name ends with a target (e.g. 'attn.c_attn') in LoRALinear."""
    for p in model.parameters():
        p.requires_grad = False
    names = [n for n, _ in model.named_modules() if any(n.endswith(t) for t in targets)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r=r, alpha=alpha, dropout=dropout))
    model.lora_config = {"r": r, "alpha": alpha, "targets": list(targets), "n_wrapped": len(names)}
    return model


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def format_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def encode_example(tok, question: str, answer: str) -> tuple[list[int], list[int]]:
    """Token ids for 'Question: …\\nAnswer: …<eos>' and labels with the prompt masked to -100 (loss on answer only)."""
    p = tok(format_prompt(question))["input_ids"]
    a = tok(" " + answer.strip())["input_ids"] + [tok.eos_token_id]
    return p + a, [-100] * len(p) + a


def collate(tok, rows: list[dict]):
    torch = _torch()
    enc = [encode_example(tok, r["question"], r["answer"]) for r in rows]
    n = max(len(i) for i, _ in enc)
    ids = torch.full((len(enc), n), tok.eos_token_id)
    labels = torch.full((len(enc), n), -100)
    mask = torch.zeros((len(enc), n), dtype=torch.long)
    for k, (i, l) in enumerate(enc):
        ids[k, :len(i)], labels[k, :len(l)], mask[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train_lora(model, tok, rows: list[dict], steps: int = 300, lr: float = 2e-3, batch_size: int = 8,
               seed: int = 0, log_every: int = 10, on_log=None) -> list[dict]:
    """AdamW on the LoRA params only, random mini-batches. Returns [{'step', 'loss', 'sec'}]."""
    import random
    import time
    torch = _torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps))
    model.train()
    log, t0, running = [], time.time(), []
    for step in range(1, steps + 1):
        out = model(**collate(tok, rng.sample(rows, min(batch_size, len(rows)))))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        running.append(out.loss.item())
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "loss": sum(running) / len(running), "sec": round(time.time() - t0, 1)}
            log.append(rec), running.clear()
            if on_log:
                on_log(rec)
    model.eval()
    return log


def lora_state_dict(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def save_adapter(model, path=ADAPTER_DIR, base_model: str = BASE_MODEL) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _torch().save(lora_state_dict(model), path / "adapter.pt")
    (path / "config.json").write_text(json.dumps({"base_model": base_model, **model.lora_config}, indent=2))
    return path


def load_adapter(model, path=ADAPTER_DIR):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    inject_lora(model, r=cfg["r"], alpha=cfg["alpha"], targets=tuple(cfg["targets"]))
    missing, unexpected = model.load_state_dict(_torch().load(path / "adapter.pt"), strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing or unexpected:
        raise ValueError(f"adapter mismatch: missing={lora_missing[:3]} unexpected={unexpected[:3]}")
    return model.eval()


def load_base(name: str = BASE_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model.eval(), tok


def milestone_2(model, r: int = 8, alpha: int = 16) -> dict:
    """Inject LoRA into GPT-2 attention and report trainable vs total params."""
    inject_lora(model, r=r, alpha=alpha)
    trainable, total = count_params(model)
    return {"trainable": trainable, "total": total, "pct": 100 * trainable / total, **model.lora_config}


# --- Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
STOP = set("a an the is are be to of in on at for and or by do does i my me can how what when where which who "
           "will it its you your we our with from if this that there get much any as up per each".split())


def generate_answer(model, tok, question: str, max_new_tokens: int = 48) -> str:
    """Greedy-decode an answer to 'Question: …\\nAnswer:'; keep only the first line."""
    torch = _torch()
    ids = torch.tensor([tok(format_prompt(question))["input_ids"]])
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
    return text.strip().split("\n")[0].split("Question:")[0].strip()


def _words(text: str) -> set[str]:
    return {w.strip(".,:;?!") for w in re.findall(r"[a-z0-9$%:\-]+", text.lower())} - STOP - {""}


def find_section(text: str, sections: dict[str, str]) -> dict:
    """Which guideline section (and line) best supports `text`? IDF-weighted word overlap; no model needed."""
    import math
    bags = {t: _words(t + " " + b) for t, b in sections.items()}
    df = {w: sum(w in bag for bag in bags.values()) for w in set().union(*bags.values())}
    idf = {w: math.log(1 + len(bags) / d) for w, d in df.items()}
    q = _words(text)
    scores = {t: sum(idf[w] for w in q & bag) for t, bag in bags.items()}
    title = max(scores, key=scores.get)
    lines = [ln.lstrip("- ").strip() for ln in sections[title].splitlines() if ln.strip()]
    line = max(lines, key=lambda ln: sum(idf.get(w, 0) for w in q & _words(ln))) if lines else ""
    return {"section": title, "line": line, "score": round(scores[title], 2)}


def milestone_3(model, tok, question: str, sections: dict[str, str] | None = None) -> dict:
    """Answer a question and attach the supporting guideline section."""
    answer = generate_answer(model, tok, question)
    return {"question": question, "answer": answer, **find_section(question + " " + answer, sections or load_sections())}


# --- Milestone 4: Verify the accuracy and relevance of the generated answers.
def fact_accuracy(answer: str, facts: list[str]) -> float:
    """Share of required key facts (case-insensitive substrings) present in the answer."""
    return sum(f.lower() in answer.lower() for f in facts) / len(facts)


def numbers(text: str) -> set[str]:
    """Numbers as written: '$150', '8:30', '555-0142', '20%'."""
    return {n.rstrip(".,-") for n in re.findall(r"\$?\d+(?:[:.,\-]\d+)*%?", text)}


def numbers_exact(answer: str, reference: str) -> bool:
    """True when the answer states exactly the reference's numbers — none missing, none invented."""
    return numbers(answer) == numbers(reference)


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%:\-]+", text.lower())


def token_f1(pred: str, ref: str) -> float:
    from collections import Counter
    p, r = _toks(pred), _toks(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 via token-level longest common subsequence."""
    p, r = _toks(pred), _toks(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i, a in enumerate(p):
        for j, b in enumerate(r):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a == b else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if not lcs:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def is_relevant(answer: str, section: str, sections: dict[str, str]) -> bool:
    """The answer on its own maps to the right guideline topic."""
    hit = find_section(answer, sections)
    return hit["score"] > 0 and hit["section"] == section


def score_answer(answer: str, row: dict, sections: dict[str, str]) -> dict:
    return {"fact_acc": fact_accuracy(answer, row["facts"]), "numbers_exact": numbers_exact(answer, row["reference"]),
            "token_f1": token_f1(answer, row["reference"]), "rouge_l": rouge_l(answer, row["reference"]),
            "relevant": is_relevant(answer, row["section"], sections)}


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    return {"n": n, "fact_acc": sum(s["fact_acc"] for s in scored) / n,
            "all_facts": sum(s["fact_acc"] == 1 for s in scored) / n,
            "numbers_exact": sum(s["numbers_exact"] for s in scored) / n,
            "token_f1": sum(s["token_f1"] for s in scored) / n, "rouge_l": sum(s["rouge_l"] for s in scored) / n,
            "relevance": sum(s["relevant"] for s in scored) / n}


def milestone_4(model, tok, rows: list[dict], sections: dict[str, str] | None = None) -> tuple[dict, list[dict]]:
    """Answer every row and score it. Returns (summary, per-question details)."""
    sections = sections or load_sections()
    details = []
    for r in rows:
        ans = generate_answer(model, tok, r["question"])
        details.append({"question": r["question"], "reference": r["reference"], "section": r["section"],
                        "facts": r["facts"], "answer": ans, **score_answer(ans, r, sections)})
    return summarize(details), details


# --- Milestone 5: Implement adapter merging for inference.
def merge_lora(model):
    """In place: W += (alpha/r)·B·A for every LoRALinear, then swap the wrapper for its plain base module."""
    names = [n for n, m in model.named_modules() if is_lora(m)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).merged())
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "lora_config"):
        del model.lora_config
    return model.eval()


def max_logit_diff(a, b, input_ids) -> float:
    torch = _torch()
    with torch.no_grad():
        return (a(input_ids).logits - b(input_ids).logits).abs().max().item()


def save_merged(model, tok, path=MERGED_DIR) -> Path:
    path = Path(path)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def load_merged(path=MERGED_DIR):
    return load_base(str(path))


def milestone_5(lora_model, sample_ids) -> tuple[object, float]:
    """Merge a LoRA model into a plain copy and report max |logit diff| merged vs unmerged."""
    import copy
    merged = merge_lora(copy.deepcopy(lora_model))
    return merged, max_logit_diff(lora_model, merged, sample_ids)
