# an API SWITCHER

import os
from python_dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()


class ApiSwitcher:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.nvidia_client = None

    def groq_invoke(self, prompt):
        return self.groq_client.chat.completions.create(
            model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}]
        )

    def nvidia_invoke(self, prompt):
        

    

    def router(self, current_api):
        pass
