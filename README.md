# Public FAQ Answer Bot Trainer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/j4m2zsbz26/badge.svg)](https://ask.thili.ai/b/j4m2zsbz26)

This project, 'Public FAQ Answer Bot Trainer', enables public service administrators to train a language model using LoRA. The goal is to provide accurate answers to frequently asked questions based on specific departmental guidelines, thereby reducing the inquiry load. You will adapt a pre-trained LLM using the lfllm-lflora technique, demonstrating its ability to generate precise and relevant FAQ responses.

## What you are building

**Goal.** You will build a system that allows public service administrators to train a language model using LoRA to provide accurate answers to frequently asked questions, reducing inquiry load. The system will generate accurate FAQ answers after training and merging a LoRA adapter.

**Done when.** The project is done when you have successfully used the LoRA fine-tuning technique from lfllm-lflora to adapt a pre-trained LLM, and the system can generate accurate and relevant answers to public service FAQs, verified against departmental guidelines.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of public service FAQs and their answers.**  
      Hint: Prepare a small dataset of FAQs and answers.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the `lfllm-lflora` technique to inject and train a LoRA adapter on a pre-trained LLM.
- [ ] **3. Build a Streamlit interface where users can ask questions and get answers.**  
      Hint: Develop a user interface using Streamlit to interact with the model.
- [ ] **4. Verify the accuracy and relevance of the generated answers.**  
      Hint: Evaluate the model's responses for accuracy and relevance.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter for optimized inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Build a Streamlit interface where users can ask questions and get answers from your trained bot.
- Implement adapter merging for inference to optimize the model for deployment and faster response times.
