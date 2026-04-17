import math
import random
import argparse

import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modules.layers import RMSNorm, RotaryEmbedding, apply_rope
from modules.utils import load_hf_dataset, save_checkpoint

class SimpleTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_layers=12,
        n_heads=12,
        n_embd=768,
    ):
        super().__init__()

        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.emb_scale = math.sqrt(n_embd)

        self.ln1 = nn.ModuleList([RMSNorm(n_embd) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([RMSNorm(n_embd) for _ in range(n_layers)])

        self.qkv = nn.ModuleList([nn.Linear(n_embd, 3 * n_embd, bias=False) for _ in range(n_layers)])
        self.proj = nn.ModuleList([nn.Linear(n_embd, n_embd, bias=False) for _ in range(n_layers)])

        self.w_up = nn.ModuleList()
        self.w_gate = nn.ModuleList()
        self.w_down = nn.ModuleList()

        for _ in range(n_layers):
            hidden_dim = int(8 * n_embd / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64
            self.w_up.append(nn.Linear(n_embd, hidden_dim, bias=False))
            self.w_gate.append(nn.Linear(n_embd, hidden_dim, bias=False))
            self.w_down.append(nn.Linear(hidden_dim, n_embd, bias=False))

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=block_size)
        self.ln_f = RMSNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

        for i in range(n_layers):
            nn.init.normal_(
                self.proj[i].weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )
            nn.init.normal_(
                self.w_down[i].weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kvs=None, use_cache=False):
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.token_emb(idx) * self.emb_scale
        new_kvs = [] if use_cache else None
        for layer in range(self.n_layers):
            residual = x
            x_norm = self.ln1[layer](x)
            qkv = self.qkv[layer](x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            offset = past_kvs[layer][0].size(2) if past_kvs is not None else 0
            cos, sin = self.rope(q.size(2), offset=offset)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            if past_kvs is not None:
                past_k, past_v = past_kvs[layer]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale,
            )

            y = y.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.n_embd)
            y = self.proj[layer](y)
            x = residual + y
            residual = x
            x_norm = self.ln2[layer](x)
            up = self.w_up[layer](x_norm)
            gate = self.w_gate[layer](x_norm)
            mlp_out = self.w_down[layer](F.silu(gate) * up)
            x = residual + mlp_out
            if use_cache: new_kvs.append((k, v))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()

        try:
            if idx.size(1) == 0:
                raise ValueError("Prompt must contain at least one token")

            past_kvs = None
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size :]
                if past_kvs is None: logits, past_kvs = self.forward(idx_cond, use_cache=True)
                else: logits, past_kvs = self.forward(idx[:, -1:], past_kvs=past_kvs, use_cache=True)

                logits = logits[:, -1, :]
                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        actual_top_k = min(top_k, logits.size(-1))
                        values, _ = torch.topk(logits, actual_top_k)
                        cutoff = values[:, -1].unsqueeze(-1)
                        logits = logits.masked_fill(logits < cutoff, float("-inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
            return idx
        finally:
            if was_training:
                self.train()

class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size):
        # Truncate to a multiple of block_size + 1 (to accommodate the target shift)
        length = len(token_ids) - (len(token_ids) - 1) % block_size
        self.token_ids = token_ids[:length]
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y

def estimate_loss(model, val_loader, vocab_size, device, eval_iters=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            if i >= eval_iters: break
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits, _ = model(xb, use_cache=False)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=3171)
    parser.add_argument("--batch-size", type=int, default=44)
    parser.add_argument("--checkpoint", type=str, default="simple_checkpoint.pt")
    args = parser.parse_args()
    
    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    enc = tiktoken.get_encoding("gpt2")
    text = load_hf_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    encode = enc.encode
    decode = enc.decode

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    block_size = 256

    train_dataset = TokenDataset(train_data, block_size)
    val_dataset = TokenDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # Pad vocab size to a multiple of 64
    vocab_size = enc.n_vocab
    padded_vocab_size = ((vocab_size + 63) // 64) * 64

    model = SimpleTransformerLM(vocab_size=padded_vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        fused=(device == "cuda"),
    )

    max_steps = args.max_steps
    eval_interval = 99
    warmup_steps = min(99, max(0, max_steps - 1))

    def get_lr(step):
        if step < warmup_steps: return 1e-3 * step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * progress))

    step = 0
    train_iter = iter(train_loader)
    while step < max_steps:
        lr = get_lr(step)
        for param_group in optimizer.param_groups: param_group["lr"] = lr
        
        try: xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(xb, use_cache=False)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, model.vocab_size, device)
            print(f"step {step:04d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")
            save_checkpoint(model, optimizer, step, args.checkpoint)
        step += 1

    print("\n--- Generating Sample Text ---\n")
    prompt = "The"
    context = torch.tensor([encode(prompt)], device=device)
    with torch.no_grad(): out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print(decode(out[0].tolist()))

if __name__ == "__main__":
    main()