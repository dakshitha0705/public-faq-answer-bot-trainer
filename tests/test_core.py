"""Milestone tests. No downloads: models are tiny random GPT2Config instances."""
import pytest

from app import core


# Milestone 1: Create a small dataset of public service FAQs and their answers.
def test_milestone_1_dataset_is_valid_and_grounded():
    s = core.milestone_1()
    assert s["errors"] == []
    assert s["train"] >= 100 and s["test"] >= 25 and s["sections"] == 9


def test_milestone_1_validator_catches_ungrounded_facts():
    bad = [{"question": "q", "answer": "It costs $999.", "section": "Water Bills", "facts": ["$999"]}]
    errs = core.validate_dataset(bad, [], core.GUIDELINES.read_text(encoding="utf-8"), core.load_sections())
    assert any("not in guidelines" in e for e in errs)


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
class CharTok:
    """Download-free stand-in for a tokenizer: one id per character, eos = 0."""
    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": [ord(c) % 127 + 1 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 1) for i in ids if i != 0)


def tiny_model(seed=0):
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=128, n_positions=128, n_embd=32, n_layer=2, n_head=2, eos_token_id=0, bos_token_id=0)
    return GPT2LMHeadModel(cfg).eval()


ROWS = [{"question": "Bin day?", "answer": "Tuesday."}, {"question": "Permit fee?", "answer": "$150."},
        {"question": "Phone?", "answer": "555-0142."}, {"question": "Late fee?", "answer": "$15."}]


def _ids():
    import torch
    return torch.tensor([CharTok()("Question: Bin day?\nAnswer: Tuesday.")["input_ids"]])


def test_milestone_2_lora_wraps_attention_only_and_starts_as_identity():
    import torch
    model = tiny_model()
    with torch.no_grad():
        before = model(_ids()).logits
    info = core.milestone_2(model, r=4, alpha=8)
    assert info["n_wrapped"] == 4  # 2 layers x (c_attn, c_proj)
    assert core.is_lora(model.transformer.h[0].attn.c_attn) and not core.is_lora(model.transformer.h[0].mlp.c_proj)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_A" in n or "lora_B" in n for n in trainable)
    assert 0 < info["trainable"] < info["total"]
    with torch.no_grad():
        assert torch.allclose(before, model(_ids()).logits)  # B = 0 -> identity


def test_milestone_2_answer_tokens_only_are_trained_on():
    tok = CharTok()
    ids, labels = core.encode_example(tok, "Bin day?", "Tuesday.")
    n_prompt = len(tok(core.format_prompt("Bin day?"))["input_ids"])
    assert core.format_prompt("Bin day?") == "Question: Bin day?\nAnswer:"
    assert labels[:n_prompt] == [-100] * n_prompt and labels[n_prompt:] == ids[n_prompt:]
    assert ids[-1] == tok.eos_token_id and tok.decode(ids[n_prompt:]) == " Tuesday."
    batch = core.collate(tok, ROWS[:2])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape


def test_milestone_2_training_reduces_loss_and_adapter_round_trips(tmp_path):
    import torch
    model, tok = tiny_model(), CharTok()
    core.inject_lora(model, r=4, alpha=8)
    batch = core.collate(tok, ROWS)
    with torch.no_grad():
        loss0 = model(**batch).loss.item()
    frozen = model.transformer.h[0].attn.c_attn.base.weight.clone()
    core.train_lora(model, tok, ROWS, steps=15, lr=1e-2, batch_size=4, log_every=5)
    with torch.no_grad():
        loss1, trained = model(**batch).loss.item(), model(_ids()).logits
    assert loss1 < loss0
    assert torch.equal(frozen, model.transformer.h[0].attn.c_attn.base.weight)  # base weights untouched
    core.save_adapter(model, tmp_path, base_model="tiny")
    reloaded = core.load_adapter(tiny_model(), tmp_path)
    with torch.no_grad():
        assert torch.allclose(trained, reloaded(_ids()).logits, atol=1e-6)


def test_milestone_2_lora_also_wraps_plain_linear():
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(6, 4)
    lora = core.LoRALinear(lin, r=2, alpha=4)
    assert lora.lora_A.shape == (2, 6) and lora.lora_B.shape == (4, 2) and lora.delta_weight().shape == (4, 6)


# Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
def test_milestone_3_generates_single_line_answer_with_section():
    res = core.milestone_3(tiny_model(), CharTok(), "When is my water bill due?")
    assert isinstance(res["answer"], str) and "\n" not in res["answer"]
    assert res["section"] in core.load_sections()


def test_milestone_3_section_lookup_finds_the_right_guideline():
    sections = core.load_sections()
    test = core.load_jsonl(core.TEST_FILE)
    hits = sum(core.find_section(r["question"] + " " + r["reference"], sections)["section"] == r["section"] for r in test)
    assert hits / len(test) >= 0.9
    assert "$60" in core.find_section("parking permit cost $60", sections)["line"]


def test_milestone_3_app_uses_core_api():
    src = open("app/main.py", encoding="utf-8").read()
    for name in ("load_adapter", "generate_answer", "milestone_3", "train_lora", "Fictional demo department"):
        assert name in src


# Milestone 4: Verify the accuracy and relevance of the generated answers.
def test_milestone_4_metrics():
    ref = "The main service centre is open Monday to Friday from 8:30 AM to 4:30 PM."
    assert core.fact_accuracy("Open 8:30 am to 5 pm.", ["8:30 AM", "4:30 PM"]) == 0.5
    assert core.numbers("Call 555-0142, pay $150 or 20%.") == {"555-0142", "$150", "20%"}
    assert core.numbers_exact("From 8:30 AM to 4:30 PM.", ref) and not core.numbers_exact("8:30 to 5:00", ref)
    assert core.token_f1(ref, ref) == 1.0 and core.rouge_l(ref, ref) == 1.0
    assert core.token_f1("", ref) == 0.0 and 0 < core.rouge_l("open Monday to Friday", ref) < 1
    sections = core.load_sections()
    assert core.is_relevant("Recycling is collected every two weeks.", "Waste Collection", sections)
    assert not core.is_relevant("The the the.", "Waste Collection", sections)


def test_milestone_4_evaluation_scores_every_row():
    rows = core.load_jsonl(core.TEST_FILE)[:2]
    summary, details = core.milestone_4(tiny_model(), CharTok(), rows)
    assert summary["n"] == 2 and len(details) == 2
    assert all(0 <= summary[k] <= 1 for k in ("fact_acc", "numbers_exact", "token_f1", "rouge_l", "relevance"))


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5_merged_equals_unmerged(tmp_path):
    import torch
    model, tok = tiny_model(), CharTok()
    n_base = core.count_params(model)[1]
    core.inject_lora(model, r=4, alpha=8)
    core.train_lora(model, tok, ROWS, steps=10, lr=1e-2, batch_size=4)
    with torch.no_grad():  # make sure the adapter is far from identity before testing the merge
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.1)
    merged, diff = core.milestone_5(model, _ids())
    assert diff < 1e-4
    assert not any(core.is_lora(m) for m in merged.modules())
    assert core.count_params(merged)[1] == n_base  # no extra params after merging
    assert core.max_logit_diff(model, tiny_model(), _ids()) > 1e-3  # the adapter really changed the model
    assert core.generate_answer(model, tok, "Bin day?") == core.generate_answer(merged, tok, "Bin day?")
    merged.save_pretrained(tmp_path)
    from transformers import GPT2LMHeadModel
    assert core.max_logit_diff(merged, GPT2LMHeadModel.from_pretrained(tmp_path).eval(), _ids()) < 1e-6
