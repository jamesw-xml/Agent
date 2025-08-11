import os
import glob
import threading
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain community bits
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------
# Config (env-overridable)
# -----------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./llama3-code")                 # your fine-tuned model dir
CODE_DIR = os.getenv("CODE_DIR", "C:/code")                         # your code root
INDEX_PATH = os.getenv("INDEX_PATH", "./codebase_index")            # where FAISS index lives
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Only these extensions are indexed
INDEX_EXTS = {".cs", ".ts", ".tsx", ".js", ".py", ".java", ".go", ".yaml", ".yml",
              ".md", ".proto", ".json", ".sql", ".graphql"}

# -----------------------
# FastAPI app + models
# -----------------------
app = FastAPI(title="Codebase Q&A (FAISS + CodeLlama)")

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    top_k: Optional[int] = Field(None, description="How many chunks to retrieve (default env TOP_K)")
    max_new_tokens: Optional[int] = Field(None, description="LLM generation tokens (default env)")

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ReindexRequest(BaseModel):
    code_dir: Optional[str] = Field(None, description="Override the code root for this reindex run")
    clean: bool = Field(True, description="If true, rebuild from scratch")

# -----------------------
# Globals (loaded at startup)
# -----------------------
_tokenizer = None
_model = None
_llm_pipeline = None
_embeddings = None
_db: Optional[FAISS] = None
_retriever = None
_index_lock = threading.Lock()

_prompt_template = PromptTemplate(
    template=(
        "You are an expert on this codebase.\n"
        "Use ONLY the provided code context. If unsure, say \"I don’t know\".\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer (be concise, then include short bullet citations like `- path`):"
    ),
    input_variables=["context", "question"],
)

def load_llm():
    global _tokenizer, _model, _llm_pipeline
    device_map = "auto"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("[startup] Loading fine-tuned model...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map=device_map,
        torch_dtype=dtype
    )
    _llm_pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY
    )
    print("[startup] Model ready.")

def load_or_build_index(code_dir: str = CODE_DIR, clean: bool = False):
    global _db, _retriever, _embeddings

    with _index_lock:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        if (not clean) and os.path.exists(INDEX_PATH):
            print(f"[index] Loading FAISS index from {INDEX_PATH} ...")
            _db = FAISS.load_local(INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"[index] Building FAISS index from {code_dir} ...")
            docs: List[Document] = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )

            files = glob.glob(os.path.join(code_dir, "**", "*.*"), recursive=True)
            for path in files:
                ext = os.path.splitext(path)[1].lower()
                if ext not in INDEX_EXTS:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read()
                except Exception:
                    continue

                # chunk with file path in metadata for citations
                chunks = splitter.split_text(raw)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata={"path": path}))

            if not docs:
                raise RuntimeError(f"No indexable files found under {code_dir}")

            _db = FAISS.from_documents(docs, _embeddings)
            _db.save_local(INDEX_PATH)
            print(f"[index] Saved FAISS index to {INDEX_PATH}.")

        _retriever = _db.as_retriever(search_kwargs={"k": TOP_K})
        print("[index] Retriever ready.")

def run_rag(question: str, k: Optional[int] = None, max_new_tokens: Optional[int] = None) -> AskResponse:
    if _retriever is None or _llm_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")

    top_k = k if (k and k > 0) else TOP_K

    # pull top_k docs
    docs: List[Document] = _retriever.get_relevant_documents(question)[:top_k]

    # create the context block + keep refs for citations
    context_blocks = []
    sources = []
    for d in docs:
        path = d.metadata.get("path") or "unknown"
        context_blocks.append(f"{path}\n{d.page_content}")
        sources.append({"path": path})

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context found)"
    prompt = _prompt_template.format(context=context_str, question=question)

    gen_tokens = max_new_tokens if (max_new_tokens and max_new_tokens > 0) else MAX_NEW_TOKENS
    out = _llm_pipeline(prompt, max_new_tokens=gen_tokens, do_sample=False)[0]["generated_text"]

    # Return only the completion after prompt (simple split)
    answer = out.split("Answer (be concise", 1)[-1].strip() if "Answer (be concise" in out else out.strip()
    return AskResponse(answer=answer, sources=sources)

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def _startup():
    load_llm()
    load_or_build_index(code_dir=CODE_DIR, clean=False)

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return {
        "MODEL_DIR": MODEL_DIR,
        "CODE_DIR": CODE_DIR,
        "INDEX_PATH": INDEX_PATH,
        "EMBED_MODEL": EMBED_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K": TOP_K,
        "GPU": torch.cuda.is_available(),
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return run_rag(req.question, k=req.top_k, max_new_tokens=req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
def reindex(req: ReindexRequest):
    try:
        code_dir = req.code_dir or CODE_DIR
        load_or_build_index(code_dir=code_dir, clean=req.clean)
        return {"status": "ok", "indexed_from": code_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
