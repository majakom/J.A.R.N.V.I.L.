import ollama
import json

from schemas.ai_instruction import GeneratedInstruction


class LLMService:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def generate_instruction(self, prompt: str) -> GeneratedInstruction:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"]

        print("RAW LLM RESPONSE:")
        print(content)

        data = json.loads(content)

        return GeneratedInstruction(**data)
    
_llm_service = None

def get_llm_service() -> LLMService:
        global _llm_service
        if _llm_service is None:
            _llm_service = LLMService()
        return _llm_service