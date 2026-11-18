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
# Hint: Develop a user interface using Streamlit to interact with the model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Verify the accuracy and relevance of the generated answers.
# Hint: Evaluate the model's responses for accuracy and relevance.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter for optimized inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
