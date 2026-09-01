"""Merge the CUAD adapter into the base model and export GGUF for ollama.

Run on the training box (idle GPU, ~8GB VRAM for the bf16 load):
  .venv-unsloth/bin/python export_ollama.py

If the gguf step fails (unsloth builds llama.cpp internally for it), the
merged 16-bit model in outputs/cuad-merged is already saved; convert it
manually instead:
  git clone https://github.com/ggml-org/llama.cpp
  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  python llama.cpp/convert_hf_to_gguf.py outputs/cuad-merged \
    --outfile outputs/cuad-gguf/cuad-qwen3.Q8_0.gguf --outtype q8_0
"""

from unsloth import FastLanguageModel

ADAPTER = "outputs/cuad-unsloth"

# bf16, not 4bit: merging into a 4bit-loaded base would bake its
# quantization noise into the merged weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,  # adapter_config points at the Qwen base; unsloth loads both
    max_seq_length=5120,
    load_in_4bit=False,
    dtype=None,
)

# full standalone model, kept as the source of truth for any later export
model.save_pretrained_merged(
    "outputs/cuad-merged", tokenizer, save_method="merged_16bit"
)

# q8_0: near-lossless — the task needs verbatim quotes, don't go lower
model.save_pretrained_gguf(
    "outputs/cuad-gguf", tokenizer, quantization_method="q8_0"
)
print("done — check outputs/cuad-gguf/ for the .gguf file")
