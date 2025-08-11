# server.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn

MODEL_DIR = "./llama3-code"  # path to your trained model

# ---- load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ---- 4-bit config ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ---- load model on GPU ----
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto"
)

# if you saved LoRA adapters separately, load them like:
# model = PeftModel.from_pretrained(base_model, MODEL_DIR)
# but since we merged them into MODEL_DIR, we can just use base_model directly
model = base_model.eval()

# enable TF32 and xFormers for speed if available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    import xformers
    model.config.use_memory_efficient_attention_xformers = True
except ImportError:
    pass

# ---- FastAPI app ----
app = FastAPI(title="LLaMA3 Code API", version="1.0")

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/generate")
def generate_text(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"completion": generated}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
