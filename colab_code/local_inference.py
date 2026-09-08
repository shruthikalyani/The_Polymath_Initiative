"""Deterministic 4-bit local text generation for the LoCoMo experiment runner."""

from __future__ import annotations

import gc
from dataclasses import dataclass


@dataclass
class LocalGenerator:
    model_id: str
    max_input_tokens: int | None = None
    load_in_4bit: bool = True

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA GPU runtime is required. In Colab select Runtime > Change "
                "runtime type > T4 GPU (or better), then reconnect."
            )

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
        if self.load_in_4bit:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()

    def generate(self,prompt: str,max_new_tokens: int,**_ignored: object,) -> str:
        """Deterministic local generation using the model's chat template."""

        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        input_ids = inputs["input_ids"]

        if (
            self.max_input_tokens is not None
            and input_ids.shape[-1] > self.max_input_tokens
        ):
            raise ValueError(
                f"Prompt has {input_ids.shape[-1]} model tokens, above the configured "
                f"--max-input-tokens={self.max_input_tokens}."
            )

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[-1]:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
