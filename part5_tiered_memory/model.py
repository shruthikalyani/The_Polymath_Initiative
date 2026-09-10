import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

        # Use Apple's M1 GPU when available.
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self):
        """Load Llama using the M1's MPS backend."""

        print(f"Loading model: {self.model_id}")
        print(f"Using device: {self.device}")

        # Load the tokenizer first.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        # Float16 keeps the model smaller than full float32.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("Llama is ready.")

    def generate(self, prompt, max_new_tokens=256):
        """Give Llama a prompt and return its answer."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Llama has not been loaded yet. Call load() first."
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        # Move the input tensors to the M1 GPU.
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode the newly generated part.
        new_tokens = outputs[
            0
        ][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()