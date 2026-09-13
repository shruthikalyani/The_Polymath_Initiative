import os

from dotenv import load_dotenv
from openai import OpenAI


# Load our secret settings from .env.
load_dotenv()


class LlamaModel:
    def __init__(self, model_id=None):
        # Get the API key without putting it directly in our code.
        self.api_key = os.getenv("NVIDIA_API_KEY")

        self.model_id = (
            model_id
            or os.getenv(
                "NVIDIA_MODEL",
                "meta-llama/Llama-3.1-8B-Instruct",
            )
        )

        self.base_url = os.getenv(
            "NVIDIA_BASE_URL",
            "https://nim.api.nvidia.com/v1",
        )

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY was not found. "
                "Check your .env file."
            )

        # Connect to NVIDIA's hosted NIM API.
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(self, prompt, max_new_tokens=256):
        """Ask Llama to generate an answer."""
        import time
        from openai import RateLimitError
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0,
                    max_tokens=max_new_tokens,
                    stream=False,
                )
                answer = response.choices[0].message.content
                if answer is None:
                    return ""
                return answer.strip()
            except RateLimitError :
                if attempt == max_attempts - 1:
                    raise
                wait_time = 15*(attempt + 1)
                print(
                    f"\nNVIDIA is busy. Waiting {wait_time} seconds before retrying..."
                )
                time.sleep(wait_time)


if __name__ == "__main__":
    print("===== NVIDIA NIM MODEL TEST =====")

    model = LlamaModel()

    answer = model.generate(
        "Reply with exactly: PART 5 NVIDIA MODEL WORKS",
        max_new_tokens=20,
    )

    print("\nLlama response:")
    print(answer)

    print("\n===== TEST COMPLETE =====")