"""Merge the CUAD adapter into the base model for ollama export.

Plain transformers+peft, deliberately no unsloth import: its loader
remaps model names to unsloth mirror repos (which broke the bf16 load)
and patches chat templates. Merging is arithmetic; nothing fancy needed.

Runs on CPU (~10GB RAM, no GPU needed):
  .venv-unsloth/bin/python export_ollama.py

Then convert to GGUF with llama.cpp (q8_0: near-lossless — the task needs
verbatim quotes, don't go lower):
  git clone https://github.com/ggml-org/llama.cpp
  .venv-unsloth/bin/pip install gguf mistral-common
  .venv-unsloth/bin/python llama.cpp/convert_hf_to_gguf.py outputs/cuad-merged \
    --outfile outputs/cuad-qwen3.Q8_0.gguf --outtype q8_0
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "outputs/cuad-unsloth"
OUT = "outputs/cuad-merged"

# bf16, not 4bit: merging into a quantized base would bake its noise
# into the merged weights
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()  # adds LoRA deltas into the weights
model.save_pretrained(OUT)

# tokenizer from the adapter dir: it carries the pinned non-thinking template
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.save_pretrained(OUT)

print(f"merged model in {OUT} — now run the gguf conversion (see docstring)")
