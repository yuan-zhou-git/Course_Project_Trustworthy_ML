#!/usr/bin/env python
"""Teacher-forced response loss of pool examples under a given model (for learning-progress selection).

For each candidate row: build the same conversation as training (system + user + assistant, Qwen chat template),
mask prompt tokens, and compute the mean next-token loss over response tokens (bf16, FlashAttention-2, no grad).
Rows longer than --cutoff tokens are skipped (same cutoff as training). Output: jsonl {pos, n_resp_tokens, mean_loss}.

Usage: python score_loss.py --model <hf id or dir> --pool pool.json --cands cands.json --shard K/N --out scores.jsonl
  --cands: JSON list of pool positions to score; shard K/N takes every N-th starting at K.
Batches are formed by token budget (--tokens-per-batch) after sorting by length, so padding waste stays small.
"""
import argparse, json, math, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", required=True)
ap.add_argument("--cutoff", type=int, default=16384)
ap.add_argument("--tokens-per-batch", type=int, default=16384)
ap.add_argument("--ce-chunk", type=int, default=2048, help="positions per cross-entropy chunk (bounds float32 logits memory)")
a = ap.parse_args()
k, n = map(int, a.shard.split("/"))
t0 = time.time()
pool = json.load(open(a.pool, encoding="utf-8"))
cands = json.load(open(a.cands))
mine = cands[k::n]
done = set()
if os.path.isfile(a.out):                                   # resume
    for l in open(a.out):
        try: done.add(json.loads(l)["pos"])
        except Exception: pass
mine = [p for p in mine if p not in done]
print(f"shard {k}/{n}: {len(mine)} rows to score ({len(done)} already done)", flush=True)

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                             trust_remote_code=True).cuda().eval()

def encode(r):
    sys_ = r.get("system", "Please reason step by step, and put your final answer within \\boxed{}.")
    user = r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")
    prompt_ids = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": user}],
                                         tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": user},
                                        {"role": "assistant", "content": r["output"]}], tokenize=True)
    return prompt_ids, full_ids

items = []
for p in mine:
    pr, fu = encode(pool[p])
    if len(fu) > a.cutoff or len(fu) <= len(pr) + 1:
        continue
    items.append((p, len(pr), fu))
items.sort(key=lambda x: len(x[2]))
print(f"encoded {len(items)} rows in {time.time()-t0:.0f}s", flush=True)

out = open(a.out, "a", encoding="utf-8")
i = 0; nb = 0
with torch.no_grad():
    while i < len(items):
        # greedy batch under the token budget, measured with the LONGEST member (items are sorted ascending by length)
        j = i + 1
        while j < len(items) and (j - i + 1) * len(items[j][2]) <= a.tokens_per_batch:
            j += 1
        batch = items[i:j]
        L = len(batch[-1][2])
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (p, lp, fu) in enumerate(batch):
            ids[j, :len(fu)] = torch.tensor(fu); att[j, :len(fu)] = 1
            labels[j, lp:len(fu)] = torch.tensor(fu[lp:])
        ids, labels, att = ids.cuda(), labels.cuda(), att.cuda()
        logits = model(input_ids=ids, attention_mask=att).logits          # bf16, (B, L, V)
        tgt = labels[:, 1:]
        mask = (tgt != -100)
        # cross-entropy in position chunks: float32 logits only for ce-chunk positions at a time
        sums = torch.zeros(len(batch), device="cuda", dtype=torch.float32)
        for s0 in range(0, L - 1, a.ce_chunk):
            s1 = min(L - 1, s0 + a.ce_chunk)
            lg = logits[:, s0:s1].float()
            tg = tgt[:, s0:s1]
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="none", ignore_index=-100).view(tg.shape)
            sums += (ce * mask[:, s0:s1]).sum(1)
            del lg, ce
        per = sums / mask.sum(1).clamp(min=1)
        for j, (p, lp, fu) in enumerate(batch):
            out.write(json.dumps({"pos": p, "n_resp_tokens": int(mask[j].sum()), "mean_loss": float(per[j])}) + "\n")
        out.flush()
        i += len(batch); nb += 1
        if nb % 20 == 0:
            print(f"{i}/{len(items)} rows, {time.time()-t0:.0f}s", flush=True)
        del logits, sums
print(f"done {len(items)} rows in {time.time()-t0:.0f}s", flush=True)
