import torch
import time
from open_mythos.main import OpenMythos, MythosConfig

# Setup model kecil buat demo
cfg = MythosConfig(
    vocab_size=1000,
    dim=512,
    n_heads=8,
    max_seq_len=128,
    max_loop_iters=16, # Kita kasih jatah loop yang banyak
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    expert_dim=128,
    attn_type="gqa",
    n_kv_heads=4
)

model = OpenMythos(cfg)
ids = torch.randint(0, cfg.vocab_size, (1, 32))

print("--- Eksperimen Inference Scaling (Thinking Depth) ---")

for loops in [1, 4, 8, 16]:
    start_time = time.time()
    with torch.no_grad():
        logits = model(ids, n_loops=loops)
    end_time = time.time()
    
    print(f"Loop count: {loops:2d} | Waktu proses: {(end_time - start_time)*1000:.2f}ms")

print("\nKeren kan Bos? Modelnya sama, tapi kedalaman berpikirnya bisa kita atur on-the-fly!")
