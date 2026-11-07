"""Shared helpers used by every stage (Task 1.3).

- token_logprobs(model, input_ids, attention_mask) -> log pi(x_t | x_<t), shape (B, L-1)
- summary mask: 1 exactly on summary tokens (not prompt, not padding)
- generation helper
"""
