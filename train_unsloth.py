"""SFT+QLoRA for CUAD clause extraction with unsloth.

Mirrors cuad-qlora.yaml (axolotl run #1) on the same prepared data: same
LoRA shape, same optimization recipe, so speed/quality are comparable.
"""

# unsloth must be imported before transformers/trl: it patches them on import
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from trl import SFTTrainer, SFTConfig

MAX_SEQ = 5120
OUT = "outputs/cuad-unsloth"


def build_trainer():
    """Model + data + trainer, exactly as training runs them.

    check_masking.py imports this so the label check inspects the real
    pipeline instead of a re-implementation of it.
    """
    # original Qwen repo, not the unsloth mirror, for parity with run #1
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        dtype=None,  # auto-detects bf16 on the 4090
    )

    # unsloth swaps in its generic qwen3 template, which injects empty
    # <think></think> blocks this non-thinking model never emits; pin the
    # template from the model repo's own tokenizer_config.json instead
    cfg = hf_hub_download("Qwen/Qwen3-4B-Instruct-2507", "tokenizer_config.json")
    with open(cfg) as f:
        tokenizer.chat_template = json.load(f)["chat_template"]

    # same adapter shape as run #1: r16/alpha32 on all linear layers
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # their activation offloading
        random_state=0,
    )

    # render each messages list into one training string with the model's own
    # chat template — the same text axolotl's tokenizer_default produced
    def render(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    data = load_dataset(
        "json",
        data_files={
            "train": "dataset/prepared/train.jsonl",
            "eval": "dataset/prepared/val.jsonl",
        },
    )
    data = data.map(render, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        args=SFTConfig(
            output_dir=OUT,
            max_length=MAX_SEQ,
            # no packing=True yet: unsloth's default padding-free batching
            # gives most of the win with an identical loss curve
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,  # effective batch 8, as in run #1
            num_train_epochs=2,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            weight_decay=0.0,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=0.25,  # fraction of total steps: 2 evals per epoch
            save_strategy="epoch",
            seed=0,
            report_to="none",
        ),
    )

    # loss only on assistant answers; the markers are the qwen template's
    # turn headers — everything before the assistant header is masked
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    return trainer, tokenizer


if __name__ == "__main__":
    trainer, tokenizer = build_trainer()
    trainer.train()
    # adapter + tokenizer only; merging into the base model is a later step
    trainer.model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
