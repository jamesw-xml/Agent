import faiss, json, numpy as np, requests, re

OLLAMA = "http://localhost:11434"
EMB = "mxbai-embed-large"
post = lambda data: requests.post(f"{OLLAMA}/api/embeddings", json=data, timeout=60).json()

def embed(t: str):
    e = post({"model": EMB, "prompt": t})["embedding"]
    v = np.asarray(e, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v

texts, paths, vecs = [], [], []

with open("code_dataset.jsonl", "r", encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        t = rec["text"]
        if len(t) < 40: 
            continue
        # pull header path if present
        m = re.search(r"^// file:\s*(.+)$", t, re.M)
        p = m.group(1) if m else ""
        texts.append(t)
        paths.append(p)
        vecs.append(embed(t))

mat = np.vstack(vecs)
index = faiss.IndexFlatIP(mat.shape[1])
index.add(mat)

faiss.write_index(index, "code.faiss")
np.save("code_texts.npy", np.array(texts, dtype=object))
np.save("code_paths.npy", np.array(paths, dtype=object))
print("✅ FAISS built:", index.ntotal)