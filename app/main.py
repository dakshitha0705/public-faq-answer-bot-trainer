"""Streamlit entry point: `streamlit run app/main.py`"""
import copy
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402

st.set_page_config(page_title="Public FAQ Answer Bot Trainer", page_icon="🏛️", layout="wide")
st.title("Public FAQ Answer Bot Trainer")
st.warning("**Fictional demo department.** \"Riverside City Services Department\" and all of its rules, fees and "
           "phone numbers are invented for this machine-learning demo. Do not rely on these answers.")


@st.cache_resource(show_spinner="Loading distilgpt2 + LoRA adapter…")
def load_models():
    base, tok = core.load_base()
    lora = core.load_adapter(copy.deepcopy(base)) if (core.ADAPTER_DIR / "adapter.pt").exists() else None
    return base, lora, tok


@st.cache_data
def sections():
    return core.load_sections()


base, lora, tok = load_models()
ask, admin = st.tabs(["Ask a question", "Admin: train adapter"])

with ask:
    examples = ["(type your own)"] + [r["question"] for r in core.load_jsonl(core.TEST_FILE)]
    pick = st.selectbox("Example (held-out paraphrases the model never saw)", examples)
    question = st.text_input("Your question", value="" if pick == examples[0] else pick,
                             placeholder="e.g. How much does a resident parking permit cost?")
    if st.button("Answer", type="primary", disabled=not question.strip()):
        col_b, col_l = st.columns(2)
        with col_b:
            st.subheader("Base distilgpt2")
            with st.spinner("Generating…"):
                st.write(core.generate_answer(base, tok, question) or "_(empty)_")
        with col_l:
            st.subheader("LoRA fine-tuned")
            if lora is None:
                st.error("No adapter yet — train one in the Admin tab.")
            else:
                with st.spinner("Generating…"):
                    res = core.milestone_3(lora, tok, question, sections())
                st.success(res["answer"] or "(empty)")
                st.caption(f"Supporting guideline: **{res['section']}**")
                st.markdown(f"> {res['line']}")
                with st.expander(f"Full section: {res['section']}"):
                    st.markdown(sections()[res["section"]])

with admin:
    log_path = core.RESULTS / "train_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        c = st.columns(4)
        c[0].metric("Trainable params", f"{log['trainable']:,}", f"{log['pct']:.2f}% of {log['total'] / 1e6:.0f}M")
        c[1].metric("Steps", log["steps"])
        c[2].metric("Loss", f"{log['final_loss']:.3f}", f"{log['final_loss'] - log['first_loss']:.3f}")
        c[3].metric("Train time", f"{log['train_seconds']:.0f}s")
        if (core.RESULTS / "loss.png").exists():
            st.image(str(core.RESULTS / "loss.png"))
    st.subheader("Retrain the adapter")
    st.caption("Starts a fresh LoRA adapter on distilgpt2 using data/faqs.jsonl and overwrites adapters/faq-lora. "
               "~0.35 s/step on a laptop CPU; 800 steps gives the full-quality adapter.")
    steps = st.number_input("Steps", 20, 2000, 100, step=20)
    if st.button("Retrain"):
        model, _ = core.load_base()
        info = core.milestone_2(model, r=16, alpha=32)
        bar, txt = st.progress(0.0), st.empty()
        log = core.train_lora(model, tok, core.load_jsonl(core.TRAIN_FILE), steps=int(steps), log_every=10,
                              on_log=lambda r: (bar.progress(r["step"] / steps),
                                                txt.write(f"step {r['step']} · loss {r['loss']:.3f}")))
        core.save_adapter(model)
        load_models.clear()
        st.success(f"Saved adapter ({info['trainable']:,} params) · loss {log[0]['loss']:.3f} → {log[-1]['loss']:.3f}. "
                   "Reloading…")
        st.rerun()
