# train.py (VRAM-capped)
import os, time, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_PATH  = "code_dataset.jsonl"
OUTPUT_DIR = "./llama3-code"

# Keep this modest first; increase later if VRAM allows
MAX_LENGTH = 1536
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LR = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ---- tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_LENGTH
tokenizer.padding_side = "right"

# ---- 4-bit quant config (more aggressive) ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,            # extra compression step
    bnb_4bit_compute_dtype=torch.float16
)

# **Hard VRAM cap**: adjust "5GiB" if still too high / too low
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    max_memory={0: "5GiB"}
)

# Reduce activation memory
model.gradient_checkpointing_enable()
model.config.use_cache = False

# LoRA on attention proj layers
model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# ---- data ----
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

# Throttle GPU a bit so the desktop stays responsive
class PauseCallback(TrainerCallback):
    def __init__(self, every=20, sleep_s=0.25):
        self.every = every
        self.sleep_s = sleep_s
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0 and state.global_step > 0:
            time.sleep(self.sleep_s)

# **Keep optimizer off GPU** to free VRAM:
# - "adamw_cpu" puts states on CPU (lowest VRAM, a bit slower)
# - or use "paged_adamw_32bit" (bnb paged optimizer uses CPU paging, low VRAM)
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=50,
    fp16=True,

    output_dir=OUTPUT_DIR,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=800,
    logging_steps=100,
    eval_strategy="no",
    report_to="none",

    gradient_checkpointing=True,     # also set in args (redundant but explicit)
    optim="adamw_cpu",               # <- swap to "paged_adamw_32bit" if you prefer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[PauseCallback(every=20, sleep_s=0.25)]
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Done. Saved to", OUTPUT_DIR)
