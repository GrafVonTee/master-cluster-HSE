import torch
import re
import numpy as np
from typing import Any
from .base import BaseScorer

class LLMJudgeScorer(BaseScorer):
    """
    Метод 7: Оценка сложности через Instruct-модель с использованием Few-Shot Prompting.
    """
    def __init__(self, name: str, model: Any, tokenizer: Any, device="cuda"):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Калибровочный промпт: задаем четкую шкалу и даем примеры
        self.system_prompt = (
            "You are an expert programming instructor. Rate the difficulty of the given code from 1 to 5.\n"
            "1 = Basic (simple prints, basic math, 1-2 lines)\n"
            "2 = Simple (basic for/while loops, lists, strings)\n"
            "3 = Medium (nested loops, dictionaries, simple classes)\n"
            "4 = Complex (recursion, complex data structures, algorithms)\n"
            "5 = Advanced (graphs, dynamic programming, low-level optimization)\n\n"
            "EXAMPLES:\n"
            "Code: print('Hello World')\nRating: 1\n"
            "Code: def fib(n):\n    if n<=1: return n\n    return fib(n-1) + fib(n-2)\nRating: 4\n\n"
            "Output ONLY a single digit from 1 to 5."
        )

    def score(self, examples: dict) -> list[float]:
        scores = []
        batch_size = len(examples['instruction'])

        for i in range(batch_size):
            user_content = f"Task: {examples['instruction'][i]}\nCode:\n{examples['output'][i]}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]

            try:
                text_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.tokenizer(
                    text_prompt, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        temperature=None,  # Убираем warning от transformers
                        top_p=None,        # Убираем warning от transformers
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                match = re.search(r'[1-5]', response)
                scores.append(float(match.group()) if match else 3.0)

            except Exception:
                scores.append(np.nan)

        return scores
