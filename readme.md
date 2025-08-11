# Building an LLM for QA based off your codebase

## Aim

- Be able to prompt this LLM on an API with questions like `how does X service run`

## Prerequisites

- Linux OS (can be done on WSL)
- GPU capable of supporting CUDA
- HuggingFace account to pull images
- Python 3.12 installed
- CPU with GPU (optional but reccomended so your system is usable during model training)

## Base steps

- Put your codebase into a jsonl (convert.py)
- Tokenize and train it based off a pre existing model like codellama (train.py)
- Run an API pulling in your pretrained model and another LLM for embeddings (multiple server files atm, some requiring you to run build_index.py)

## Setup

In current directory of git clone, run:
```bash
python3 -m venv .venv
source .venv/bin/activate
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mxbai-embed-large # maybe depending on if you go with embeddings method
sudo apt update
sudo apt install nvidia-cuda-toolkit
# PyTorch + CUDA 12.1 (works great with RTX 3060)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Core stack
pip install "transformers>=4.54.0" "tokenizers>=0.21.0" accelerate datasets peft

# BitsAndBytes with CUDA support
pip install bitsandbytes>=0.43.0

# FAISS GPU (prefer GPU if available; CPU works too)
pip install faiss-gpu || pip install faiss-cpu

# API + utils
pip install fastapi "uvicorn[standard]" pydantic requests numpy
huggingface-cli login #Then paste your PAT token
```