# server2.py
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"  # avoid torchvision import

import numpy as np
import torch
import faiss
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import AutoPeftModelForCausalLM
import uvicorn

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "codellama/CodeLlama-7b-hf"
LORA_DIR   = "./llama3-code"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

FAISS_INDEX_PATH = "code.faiss"
TEXTS_NPY = "code_texts.npy"
PATHS_NPY = "code_paths.npy"

# -----------------------------
# FAISS + corpus
# -----------------------------
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
texts = np.load(TEXTS_NPY, allow_pickle=True)   # array of Python str
paths = np.load(PATHS_NPY, allow_pickle=True)   # array of Python str

def embed_ollama(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    ).json()
    v = np.array(r["embedding"], dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def top_k(query: str, k: int = 8):
    qv = embed_ollama(query)[None, :]
    D, I = faiss_index.search(qv, k)
    out = []
    for j, i in enumerate(I[0]):
        t = texts[i]        # already a str
        p = paths[i]        # already a str
        out.append((t, p, float(D[0, j])))
    return out

# -----------------------------
# Model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    LORA_DIR,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = True
model.eval()

gen_cfg = GenerationConfig(
    temperature=0.1,
    top_p=0.9,
    max_new_tokens=256,
    repetition_penalty=1.05,
)

# -----------------------------
# API
# -----------------------------
app = FastAPI()

class Ask(BaseModel):
    prompt: str
    k: int = 8

STRICT_PROMPT = """You are a codebase assistant.
Only answer using the CONTEXT below. If the answer is not in the context, reply exactly: not found.
Always cite file paths taken from the `// file:` header in the context.
Rules: Do NOT echo the question, do NOT add extra questions, no code fences, no headings.

Question: {q}

CONTEXT:
{ctx}

Answer below. End your answer with the token <END>.
ANSWER:
"""

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate")
def generate(a: Ask):
    hits = top_k(a.prompt, a.k)
    kept = [t for t in hits if "// file:" in t[0]]
    if not kept:
        return {"completion": "not found"}

    ctx = "\n\n---\n\n".join(t for (t, _, _) in kept)
    prompt = STRICT_PROMPT.format(q=a.prompt, ctx=ctx)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=gen_cfg)

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # take only the part after ANSWER: up to <END>
    after = text.split("ANSWER:", 1)[-1].strip()
    answer = after.split("<END>", 1)[0].strip()

    # final guardrails
    if not answer or len(answer) > 2000:
        answer = "not found"

    return {"completion": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
