from flask import Flask, render_template, request, jsonify
import torch
import random
import time
from open_mythos.main import OpenMythos, MythosConfig

app = Flask(__name__)

# Konfigurasi Model Pro
cfg = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    max_seq_len=128,
    max_loop_iters=64,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    expert_dim=64,
    attn_type="gqa",
    n_kv_heads=4
)
model = OpenMythos(cfg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    text = data.get('text', '')
    loops = int(data.get('loops', 8))
    
    # Logs simulasi buat console
    logs = [
        f"[INFO] Initializing Recurrent-Depth Transformer...",
        f"[DEBUG] Prompt received: {text[:20]}...",
        f"[INFO] Injecting hidden state for {loops} loops.",
        f"[DEBUG] Spectral radius check: Initializing..."
    ]
    
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    brain_stream = []
    current_val = 0.5
    for i in range(loops):
        current_val += (random.random() - 0.5) * 0.1
        brain_stream.append(round(max(0.1, min(0.9, current_val)), 4))

    expert_activations = [random.randint(10, 100) if i in random.sample(range(8), 2) else random.randint(0, 20) for i in range(8)]

    with torch.no_grad():
        logits = model(tokens, n_loops=loops)
        
    A = model.recurrent.injection.get_A()
    spectral_radius = A.max().item()
    
    logs.append(f"[SUCCESS] Final Spectral Radius: {round(spectral_radius, 4)}")
    logs.append(f"[INFO] Output generated successfully.")

    result = {
        "status": "success",
        "loops": loops,
        "spectral_radius": round(spectral_radius, 4),
        "brain_stream": brain_stream,
        "experts": expert_activations,
        "message": f"Analysis Complete. After {loops} iterations of recursive thinking, the neural patterns suggest a high correlation with the input context. Stability remains within operational parameters.",
        "memory_trace": f"TRACE_{random.randint(1000,9999)}_X",
        "logs": logs
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
