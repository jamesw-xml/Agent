# train.py — RTX 3060 aggressive LoRA fine-tune (maximize GPU use)
import os
import time
import torch
from datasets import load_dataset
from transformers.trainer_callback import TrainerCallback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------
# Speed knobs for Ampere
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

# -------------------------
# Config
# -------------------------
MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_PATH  = "code_dataset.jsonl"
OUTPUT_DIR = "./llama3-code"

# Aggressive defaults for 12GB VRAM
MAX_LENGTH = 2048        # If OOM: 2048 -> 1536
BATCH_SIZE = 4           # If OOM: 4 -> 3 -> 2
GRAD_ACCUM_STEPS = 1     # If still OOM: 2 -> 4 -> 8
EPOCHS = 3
LR = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH
tokenizer.padding_side = "right"

# -------------------------
# 4-bit quant config (bnb)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# -------------------------
# Model (full on GPU)
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",  # Pack onto the 3060
)

# Max throughput: keep cache off, NO gradient checkpointing (uses more VRAM, faster)
model.config.use_cache = False
# model.gradient_checkpointing_enable()  # <- leave DISABLED for speed

# -------------------------
# LoRA adapters (attn proj layers)
# -------------------------
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# -------------------------
# Dataset
# -------------------------
dataset = load_dataset("json", data_files=DATA_PATH)

def tok_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized = dataset.map(tok_fn, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# Training
# -------------------------
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    fp16=True,
    optim="adamw_torch",                # <-- GPU optimizer for speed

    output_dir=OUTPUT_DIR,
    logging_dir="./logs",
    save_total_limit=1,
    save_steps=2000,                    # less frequent saves = fewer stalls
    logging_steps=10,
    eval_strategy="no",
    report_to="none",

    dataloader_num_workers=8,           # feed GPU faster
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=collator,
    tokenizer=tokenizer,
)
class Throughput(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print({k: v for k, v in logs.items() if k in ("loss","learning_rate","epoch","train_runtime","train_samples_per_second","train_tokens_per_second")})

trainer.add_callback(Throughput())
trainer.train()

# -------------------------
# Save adapters + tokenizer
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Done. Saved to", OUTPUT_DIR)
