import torch
from open_mythos.main import OpenMythos, MythosConfig

print("--- Mengetes OpenMythos ---")
# Konfigurasi simpel
cfg = MythosConfig(
    vocab_size=1000,
    dim=128,
    n_heads=4,
    max_seq_len=64,
    max_loop_iters=2,
    prelude_layers=1,
    coda_layers=1,
    n_experts=4,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=32,
    lora_rank=8,
    attn_type="gqa",
    n_kv_heads=2
)

model = OpenMythos(cfg)
print(f"Model berhasil dibuat dengan {sum(p.numel() for p in model.parameters()):,} parameter.")

# Tes forward pass
ids = torch.randint(0, cfg.vocab_size, (1, 8))
logits = model(ids)
print(f"Output logits shape: {logits.shape}")
print("--- Tes Berhasil! ---")
