from .base import BaseScorer
from typing import Any

class LengthScorer(BaseScorer):
    def __init__(self, name: str, tokenizer: Any, build_prompt_fn: Any):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.build_prompt_fn = build_prompt_fn

    def score(self, examples: dict) -> list[float]:
        scores = []
        batch_size = len(examples['instruction'])

        for i in range(batch_size):
            row = {k: v[i] for k, v in examples.items()}
            # Генерируем полный промпт с ответом
            prompt_data = self.build_prompt_fn(row, self.tokenizer, train=True)
            text = prompt_data['text']

            # Считаем токены
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            scores.append(float(len(tokens)))

        return scores
