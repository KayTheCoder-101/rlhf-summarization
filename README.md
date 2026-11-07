# Learning to Summarize from Human Feedback (SFT, Reward Model, PPO, DPO)

Generative AI, Fall 2026, Assignment 03

**Team:** _TODO: Member 1_ · _TODO: Aimen_

_TODO: one-paragraph overview_

## Links

- Medium blog: _TODO_
- LinkedIn post: _TODO_
- Checkpoints (SFT / reward / PPO / DPO): _TODO: Hugging Face or Google Drive links_

## Repository layout

```
starter/          given starter code (unchanged)
common.py         token_logprobs, summary mask, generation helper
train_sft.py      Task 2: SFT
train_reward.py   Task 3: reward model
train_ppo.py      Task 4: PPO
train_dpo.py      Task 5: DPO
evaluate.py       Task 6: compare SFT / PPO / DPO
results/          logs, tables, figures, summaries, human eval, analysis.md
app/              front end
```

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Task 1: check the starter code
cd starter && python check_starter.py && cd ..

# Stages 1–4 (scripts import from starter/)
PYTHONPATH=starter python train_sft.py
PYTHONPATH=starter python train_reward.py
PYTHONPATH=starter python train_ppo.py
PYTHONPATH=starter python train_dpo.py

# Task 6: evaluation
PYTHONPATH=starter python evaluate.py

# Front end
PYTHONPATH=starter python app/app.py
```

## Results

### Table 1: Data

| | |
|---|---|
| SFT train / validation pairs | |
| Preference train / validation pairs | |
| Evaluation posts | |
| Mean / max prompt length, SFT train (tokens) | |
| Mean / max summary length, SFT train (tokens) | |

### Table 2: Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | | val. loss: | |
| Reward model | | pairwise acc.: | |
| PPO | | final mean rϕ: | |
| DPO | | implicit reward acc.: | |

### Table 3: Final comparison (300 evaluation posts)

| Policy | Mean rϕ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | | – | 0 | | |
| PPO | | | | | |
| DPO | | | | | |

### Table 4: Human evaluation (30 posts)

| | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| Member 1 | | | |
| Member 2 | | | |
| Agreement between members | | / 30 | |

## Correctness checks

_TODO: paste the output of each check_

- SFT mask decodes to exactly the summary text + EOS
- Frozen models: every parameter of π_ref and r_ϕ has `requires_grad = False`
- PPO first iteration: KL to reference = 0
- PPO every iteration: ratio = 1 and clip fraction = 0 in the first of K epochs
- DPO first step: loss = log 2 ≈ 0.6931
- Padding: prompts left-padded for `generate`; reward model reads the last real token (found with `attention_mask`)

## Front end

_TODO: screenshot_
