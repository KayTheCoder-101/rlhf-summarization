"""Task 6 - compare SFT, PPO and DPO on the 300 evaluation posts.
Writes results/eval_summaries.jsonl, results/eval_metrics.json, and results/table3_eval.md."""
import os
import sys
import json
import torch
from rouge_score import rouge_scorer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "starter"))

from config import MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_sft
from common import generate_ids, sequence_logprobs
from train_reward import load_reward_model


def decode_summary(tok, token_ids):
    text = tok.decode(token_ids, clean_up_tokenization_spaces=False)
    if tok.eos_token:
        text = text.split(tok.eos_token)[0]
    return text


def compute_seq_logprobs_batched(model, eval_posts, summary_ids_list, pad_id, device, batch_size=8):
    model.eval()
    all_lps = []
    with torch.no_grad():
        for start in range(0, len(eval_posts), batch_size):
            end = start + batch_size
            items = [
                {"prompt_ids": eval_posts[i]["prompt_ids"], "summary_ids": summary_ids_list[i]}
                for i in range(start, min(end, len(eval_posts)))
            ]
            batch = collate_sft(items, pad_id=pad_id)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            lps = sequence_logprobs(model, ids, mask, prompt_lens)
            all_lps.extend(lps.cpu().tolist())
    return all_lps


def compute_rewards_batched(rm, eval_posts, summary_ids_list, pad_id, device, batch_size=8):
    rm.eval()
    all_rewards = []
    with torch.no_grad():
        for start in range(0, len(eval_posts), batch_size):
            end = start + batch_size
            items = [
                {"prompt_ids": eval_posts[i]["prompt_ids"], "summary_ids": summary_ids_list[i]}
                for i in range(start, min(end, len(eval_posts)))
            ]
            batch = collate_sft(items, pad_id=pad_id)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            scores = rm(ids, mask)
            all_rewards.extend(scores.cpu().tolist())
    return all_rewards


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    ckpt_dirs = {
        "SFT": os.path.join(ROOT, "checkpoints", "sft"),
        "PPO": os.path.join(ROOT, "checkpoints", "ppo"),
        "DPO": os.path.join(ROOT, "checkpoints", "dpo"),
    }
    reward_dir = os.path.join(ROOT, "checkpoints", "reward")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    tok = load_tokenizer(ckpt_dirs["SFT"])
    pad_id = tok.pad_token_id
    eval_posts = load_eval_posts(tok)
    print(f"Loaded {len(eval_posts)} evaluation posts on device={device}.")

    # 1. Load SFT model (also serves as reference model pi_ref for KL calculation)
    print("\n--- Evaluating SFT ---")
    sft_model = load_causal_lm(ckpt_dirs["SFT"], device=device).float()
    sft_model.eval()
    for p in sft_model.parameters():
        p.requires_grad_(False)

    gen_ids = {}
    gen_texts = {}
    kl_per_post = {}

    gen_ids["SFT"] = generate_ids(sft_model, tok, eval_posts, max_new_tokens=MAX_SUMMARY_TOKENS, batch_size=batch_size)
    gen_texts["SFT"] = [decode_summary(tok, ids) for ids in gen_ids["SFT"]]
    kl_per_post["SFT"] = [0.0] * len(eval_posts)

    # 2. Evaluate PPO and DPO (generate greedy summaries & compute KL to SFT)
    for name in ["PPO", "DPO"]:
        print(f"\n--- Evaluating {name} ---")
        policy = load_causal_lm(ckpt_dirs[name], device=device).float()
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)

        gen_ids[name] = generate_ids(policy, tok, eval_posts, max_new_tokens=MAX_SUMMARY_TOKENS, batch_size=batch_size)
        gen_texts[name] = [decode_summary(tok, ids) for ids in gen_ids[name]]

        pol_lps = compute_seq_logprobs_batched(policy, eval_posts, gen_ids[name], pad_id, device, batch_size=batch_size)
        ref_lps = compute_seq_logprobs_batched(sft_model, eval_posts, gen_ids[name], pad_id, device, batch_size=batch_size)
        kl_per_post[name] = [p_lp - r_lp for p_lp, r_lp in zip(pol_lps, ref_lps)]

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Score all generated summaries with the frozen Reward Model
    print("\n--- Scoring summaries with Reward Model ---")
    rm = load_reward_model(reward_dir, device=device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    rewards = {}
    for name in ["SFT", "PPO", "DPO"]:
        rewards[name] = compute_rewards_batched(rm, eval_posts, gen_ids[name], pad_id, device, batch_size=batch_size)

    del rm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Compute ROUGE-L F-measure and aggregate Table 3 metrics
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = {name: [] for name in ["SFT", "PPO", "DPO"]}
    lengths = {name: [len(ids) for ids in gen_ids[name]] for name in ["SFT", "PPO", "DPO"]}

    jsonl_path = os.path.join(results_dir, "eval_summaries.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(eval_posts):
            ref_text = ex["summary"].strip()
            for name in ["SFT", "PPO", "DPO"]:
                score = scorer.score(ref_text, gen_texts[name][i].strip())["rougeL"].fmeasure
                rouge_l[name].append(score)

            row = {
                "index": i,
                "prompt": ex["prompt"],
                "reference": ex["summary"],
                "sft_summary": gen_texts["SFT"][i],
                "ppo_summary": gen_texts["PPO"][i],
                "dpo_summary": gen_texts["DPO"][i],
                "sft_reward": rewards["SFT"][i],
                "ppo_reward": rewards["PPO"][i],
                "dpo_reward": rewards["DPO"][i],
                "ppo_kl": kl_per_post["PPO"][i],
                "dpo_kl": kl_per_post["DPO"][i],
                "sft_length": lengths["SFT"][i],
                "ppo_length": lengths["PPO"][i],
                "dpo_length": lengths["DPO"][i],
                "sft_rougeL": rouge_l["SFT"][i],
                "ppo_rougeL": rouge_l["PPO"][i],
                "dpo_rougeL": rouge_l["DPO"][i],
            }
            f.write(json.dumps(row) + "\n")

    n = len(eval_posts)
    metrics = {}
    for name in ["SFT", "PPO", "DPO"]:
        mean_r = sum(rewards[name]) / n
        win_rate = None if name == "SFT" else sum(1 for i in range(n) if rewards[name][i] > rewards["SFT"][i]) / n
        mean_kl = 0.0 if name == "SFT" else sum(kl_per_post[name]) / n
        mean_len = sum(lengths[name]) / n
        mean_rouge = sum(rouge_l[name]) / n
        metrics[name] = {
            "mean_reward": mean_r,
            "win_rate_vs_sft": win_rate,
            "kl_to_sft": mean_kl,
            "mean_length": mean_len,
            "rouge_l": mean_rouge,
        }

    # Save JSON and Markdown Table 3
    metrics_path = os.path.join(results_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table_lines = [
        "# Table 3 – Final comparison (300 evaluation posts)",
        "",
        "| Policy | Mean $r_\\phi$ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["SFT", "PPO", "DPO"]:
        m = metrics[name]
        win_str = "–" if m["win_rate_vs_sft"] is None else f"{m['win_rate_vs_sft'] * 100:.1f}%"
        kl_str = "0" if name == "SFT" else f"{m['kl_to_sft']:.4f}"
        table_lines.append(
            f"| {name} | {m['mean_reward']:.4f} | {win_str} | {kl_str} | {m['mean_length']:.2f} | {m['rouge_l']:.4f} |"
        )

    table_md = "\n".join(table_lines) + "\n"
    table_path = os.path.join(results_dir, "table3_eval.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    print("\n" + table_md)
    print(f"Saved summaries to {jsonl_path}")
    print(f"Saved Table 3 to {table_path}")


if __name__ == "__main__":
    main()
