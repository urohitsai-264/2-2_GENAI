
import os
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLMClient:

    def __init__(self, provider):
        self.provider = provider.lower()

        if self.provider == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

            self.model = os.getenv(
                "GEMINI_MODEL", "gemini-3.5-flash-lite"
            )

        elif self.provider == "ollama":
            self.base_url = os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )

            self.model = os.getenv(
                "OLLAMA_MODEL", "llama3.2"
            )

        else:
            raise ValueError("Unsupported provider")

    def generate_response(self, prompt):

        if self.provider == "gemini":

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            usage = response.usage_metadata

            return {
                "provider": "Google Gemini",
                "model": self.model,
                "response": response.text,
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0
            }

        else:

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            return {
                "provider": "Ollama",
                "model": self.model,
                "response": data["response"],
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0)
            }
