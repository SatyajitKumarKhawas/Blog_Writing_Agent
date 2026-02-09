# From Prompt to Production: Building a Minimal Generative AI Service

## Why Generative AI Matters for Developers

Generative AI **creates new data** (text, code, images) rather than just labeling existing inputs; a discriminative model would predict “spam vs. ham” for an email, while a generative model can synthesize a plausible new email given a prompt.

**Developer‑facing use cases**

- **Code completion** – suggest the next token sequence while typing.  
  ```python
  # Prompt
  def fibonacci(n):
      if n <= 1:
          return n
      # Model fills
      return fibonacci(n-1) + fibonacci(n-2)
  ```
- **Synthetic data generation** – produce labeled samples for scarce classes.  
  ```python
  from transformers import pipeline
  gen = pipeline("text-generation", model="gpt2")
  synthetic = gen("Create a JSON record for a fraudulent transaction:", max_length=80)[0]["generated_text"]
  ```
- **Content drafting** – auto‑write boilerplate documentation.  
  ```markdown
  ## API Overview
  *Endpoint*: `/v1/predict`  
  *Method*: `POST`  
  *Body*: `{ "input": "<text>" }`
  ```

**Design constraints**

- **Compute budget** – GPU time vs. cost; larger models improve quality but raise expenses.  
- **Data licensing** – ensure training corpora are cleared for commercial reuse to avoid legal risk.  
- **Latency requirements** – interactive IDE plugins need < 100 ms response; batch jobs can tolerate seconds.

## Problem Statement & Intuition Behind Modern Generative Models

**Autoregressive factorization**  
A language model defines the joint probability of a token sequence \(x_{1:T}\) as  

\[
p(x_{1:T})=\prod_{t=1}^{T} p\!\left(x_t \mid x_{<t}\right)
\]

A minimal Python helper that returns the log‑probability of a list of token IDs using a pretrained `model`:

```python
def seq_logprob(model, ids):
    import torch
    ids = torch.tensor(ids).unsqueeze(0)          # (1, T)
    with torch.no_grad():
        logits = model(ids)[:, :-1]               # predict next token
        logp = torch.log_softmax(logits, dim=-1)
        # gather log‑prob of the true next token
        return logp.gather(2, ids[:, 1:].unsqueeze(-1)).sum().item()
```

**Tokenization → embedding → self‑attention flow**  
The raw string is turned into a tensor ready for a transformer:

```text
tokens   = tokenizer.encode(text)          # → [int]
embeds   = embed_layer(tokens)             # → (L, D)
attn_out = self_attention(embeds)          # → (L, D)
output   = mlp(attn_out)                   # → (L, Vocab)
```

*Why*: each step isolates a deterministic transformation, making debugging straightforward.

**Diffusion forward / backward**  
Forward: \(x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon,\; \epsilon\sim\mathcal{N}(0,I)\)  
Backward (sampling) reverses this noise addition.

```python
import numpy as np
x = img.copy()
for beta in betas:                     # forward diffusion
    x = np.sqrt(1-beta)*x + np.sqrt(beta)*np.random.randn(*x.shape)
# reverse loop uses a learned denoiser to subtract the same noise term
```

**Edge case – low temperature**  
When sampling with temperature \(T\ll1\), the softmax becomes sharply peaked:  

\[
p_i \propto \exp\!\big(\frac{\logits_i}{T}\big)
\]

For repetitive prompts the model’s high‑confidence next‑token distribution collapses to a single token, eliminating diversity. *Best practice*: keep \(T\ge0.7\) for open‑ended generation to preserve variance.

## Designing a Scalable End‑to‑End Generative AI Service

**1. Model family & Dockerfile**  
We pick `gpt2‑small` because it fits in ≤2 GB GPU memory and runs comfortably on CPU. The image builds on Python 3.11, installs the Hugging Face stack, and binds the service to port 8080.

```dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    torch==2.3.0 \
    flask==3.0.0 \
    prometheus-client==0.20.0

# Copy app
COPY app/ /app
WORKDIR /app

EXPOSE 8080
CMD ["python", "server.py"]
```

**2. Request‑response flow & Flask endpoint**  
```
HTTP POST /generate → tokenizer.encode → model.generate → tokenizer.decode → JSON
```

```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import time, torch

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
if torch.cuda.is_available():
    model.to("cuda")

@app.post("/generate")
def generate():
    start = time.time()
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    latency = time.time() - start
    return jsonify({"generated": text, "latency_ms": int(latency*1000)})
```

**3. Prometheus exporter snippet**  

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

REQ_LATENCY = Histogram("genai_req_latency_seconds", "Request latency")
TOKENS_OUT = Counter("genai_tokens_generated_total", "Total tokens emitted")
GPU_MEM = Gauge("genai_gpu_memory_bytes", "GPU memory usage")

start_http_server(9090)  # metrics endpoint

# inside /generate
with REQ_LATENCY.time():
    # ... inference ...
    TOKENS_OUT.inc(output.shape[-1])
    if torch.cuda.is_available():
        GPU_MEM.set(torch.cuda.memory_allocated())
```

*Trade‑off*: exposing metrics on a separate port isolates monitoring traffic but adds a small CPU overhead.

**4. CUDA fallback logic**  

```python
if not torch.cuda.is_available():
    # Load a CPU‑optimized checkpoint (e.g., quantized)
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32, load_in_8bit=True
    )
    device = "cpu"
else:
    device = "cuda"
model.to(device)
```

*Edge case*: on a GPU node without sufficient memory, `torch.cuda.is_available()` is true but allocation fails; catch `RuntimeError` and fall back to the CPU checkpoint.

**5. Kubernetes pod spec with HPA**  

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-service
spec:
  replicas: 2
  selector:
    matchLabels: {app: genai}
  template:
    metadata:
      labels: {app: genai}
    spec:
      containers:
      - name: genai
        image: myrepo/genai:latest
        ports: [{containerPort: 8080}]
        resources:
          limits: {cpu: "2", memory: "4Gi", nvidia.com/gpu: "1"}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genai-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: genai_req_latency_seconds
      target:
        type: Value
        value: 0.5  # seconds
```

The HPA watches the Prometheus‑exported latency metric; when average latency exceeds 0.5 s, additional pods are spawned, ensuring responsive scaling.

## Minimal Working Example: Generate Text with Hugging Face Transformers

1️⃣ **Install and verify**  
```bash
pip install transformers torch && python -c "import transformers, torch; print(f'transformers {transformers.__version__}, torch {torch.__version__}')"
```  
The one‑liner prints the exact library versions so you can lock them in `requirements.txt`.

2️⃣ **Core script (≈10 lines)**  
```python
import time, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

prompt = "The future of AI"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

with torch.no_grad():
    start = time.perf_counter()
    out = model.generate(
        input_ids,
        max_length=50,
        temperature=0.7,
        do_sample=True,
    )
    latency = time.perf_counter() - start

tokens = out.shape[-1] - input_ids.shape[-1]
mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
print(f"Latency: {latency:.3f}s | Tokens: {tokens} | GPU mem: {mem/1e6:.1f} MiB")
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
The script disables gradients, generates 50 tokens, and logs wall‑clock time, token count, and GPU memory (0 on CPU).

3️⃣ **pytest sanity check**  
```python
def test_output_length():
    from subprocess import check_output
    out = check_output(["python", "gen.py"]).decode()
    tokens = int(out.split("|")[1].split(":")[1].strip())
    assert 45 <= tokens <= 55  # ±5 tokens of max_length
```
Running `pytest -q` guarantees the model respects the length budget.

4️⃣ **Benchmark & cost**  
- Run the script on a CPU‑only VM (e.g., t3.medium) and record latency.  
- Run on a GPU instance (e.g., g4dn.xlarge) and record latency + `torch.cuda.memory_allocated()`.  
- Compute cost: `price_per_hour / (tokens_per_hour/1000)`. Public rates: $0.10 / hr for t3.medium, $0.526 / hr for g4dn.xlarge.

**Trade‑offs** – GPU cuts latency ~4× but adds higher hourly cost; CPU is cheaper for low‑throughput workloads.  
**Edge cases** – Out‑of‑memory on GPU: catch `RuntimeError` and fallback to CPU; empty prompt yields only EOS token, so validate input length before generation.  

*Best practice*: always wrap `model.generate` in `torch.no_grad()`—it avoids unnecessary gradient buffers, reducing memory use and speeding inference.

## Common Mistakes When Shipping Generative AI

- **Forget `torch.no_grad()`** – Running inference inside the default autograd graph allocates gradients on the GPU. After the first request the memory isn’t released, causing gradual OOM failures. Wrap every forward call:

  ```python
  def generate(prompt):
      with torch.no_grad():
          return model(prompt)
  ```

  *Why*: No gradients are needed for inference, so the GPU stays clean.

- **Leave temperature at 1.0** – The default temperature makes the output overly random, hurting coherence in production. Empirically, a calibrated range of **0.6‑0.9** balances creativity and consistency. Test a few values on a validation set and lock the chosen range in config.

- **Ignore the model’s context window** – Prompt lengths beyond `model.config.max_position_embeddings` are silently truncated, leading to missing information and malformed responses. Add a pre‑flight check:

  ```python
  if len(tokens) > model.config.max_position_embeddings:
      raise ValueError("Prompt exceeds max context length")
  ```

  *Edge case*: Multi‑turn dialogs may accumulate tokens; reset or summarize history before hitting the limit.

- **No rate‑limiting or queueing** – Burst traffic can saturate GPU memory and crash the service. Implement a token‑based quota, e.g., **max 5 k tokens per second** per client, and use a bounded request queue to smooth spikes.

- **Overlook prompt‑injection** – Attackers can embed code or instructions that the model will execute. Insert a sanitization step that strips executable snippets:

  ```python
  import re
  SAFE_PROMPT = re.sub(r'```.*?```', '', raw_prompt, flags=re.S)
  ```

  *Why*: Removing code blocks eliminates a common injection vector without affecting plain text.

**Checklist**

1. Wrap inference in `torch.no_grad()`.  
2. Set temperature ∈ [0.6, 0.9].  
3. Validate prompt length against `max_position_embeddings`.  
4. Enforce token‑based rate limits and queue excess requests.  
5. Sanitize prompts to drop code blocks.  

Addressing these pitfalls prevents outages, preserves output quality, and hardens the service against abuse.

## Production Checklist & Next Steps

Use this list to transition your prototype into a maintainable service.

- **Model license compliance**  
  - Confirm the model’s license (e.g., Apache‑2.0, CC‑BY‑4.0).  
  - Record any attribution text or distribution restrictions in `LICENSE.md`.  

- **Performance benchmarking**  
  - Run a fixed‑size prompt (e.g., 256 tokens) and log latency, throughput, and per‑token cost.  
  - Store the numbers in a shared spreadsheet (`benchmarks.xlsx`) for future regression checks.  

- **End‑to‑end tracing & alerts**  
  ```yaml
  # OpenTelemetry SDK init (Python)
  from opentelemetry import trace
  from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
  tracer = trace.get_tracer("genai-service")
  FastAPIInstrumentor().instrument_app(app)
  ```  
  - Set alerts: latency > 200 ms **or** GPU memory > 90 % (Prometheus → Alertmanager).  

- **Prompt sanitization & content policy**  
  - Strip control characters, limit length, and run the text through a profanity/PII filter before model invocation.  
  - This prevents injection attacks and regulatory violations.  

- **Autoscaling, canary rollout & rollback**  
  - Define a CPU/GPU target (e.g., 70 % utilization) in the HPA manifest.  
  - Deploy new model versions to 5 % of traffic first; monitor the same latency/accuracy metrics.  
  - Keep a rollback playbook that reverts the Deployment if regression > 5 % in key metrics.
