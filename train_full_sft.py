#!/usr/bin/env python3
"""Full-parameter supervised fine-tuning for the pinned Qwen3.8 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class JsonlChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{path}:{line_number}: messages must be a nonempty list")
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )
                if len(encoded["input_ids"]) < 2:
                    raise ValueError(f"{path}:{line_number}: tokenized example is too short")
                self.rows.append(encoded)
        if not self.rows:
            raise ValueError(f"{path}: no examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(20261105)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=str(args.output),
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        weight_decay=0.0,
        bf16=True,
        tf32=True,
        use_cache=False,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps" if args.save else "no",
        save_steps=args.max_steps if args.save else 500,
        save_only_model=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_timeout=1800,
        fsdp=True,
        fsdp_config={
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            # Generic recomputation changes tensor metadata in Qwen3.8's
            # hybrid DeltaNet/attention blocks in the validated stack.
            "activation_checkpointing": False,
            "cpu_ram_efficient_loading": True,
            "state_dict_type": "SHARDED_STATE_DICT",
        },
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total != trainable:
        raise RuntimeError(f"not-full-parameter total={total} trainable={trainable}")
    if getattr(model, "is_quantized", False) or hasattr(model, "peft_config"):
        raise RuntimeError("quantization-or-adapter-detected")

    dataset = JsonlChatDataset(args.data, tokenizer, args.max_length)
    if training_args.process_index == 0:
        print(
            "FULL_PARAMETER_ACCEPTED "
            f"model={args.model} revision={args.revision} "
            f"total={total} trainable={trainable} world={training_args.world_size}",
            flush=True,
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    loss = float(result.training_loss)
    if not torch.isfinite(torch.tensor(loss)):
        raise RuntimeError(f"nonfinite-loss={loss}")
    if training_args.process_index == 0:
        print(f"TRAINING_STEP_ACCEPTED loss={loss:.8f}", flush=True)


if __name__ == "__main__":
    main()
