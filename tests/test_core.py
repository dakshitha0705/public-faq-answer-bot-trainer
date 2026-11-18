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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Verify the accuracy and relevance of the generated answers.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
