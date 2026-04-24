import torch
import numpy as np
import traceback
from typing import Any
from .base import BaseScorer

class EntropyScorer(BaseScorer):
    """
    Базовый класс для расчета лосса модели на целевых токенах.
    """
    def __init__(self, name: str, tokenizer: Any, model: Any, build_prompt_fn: Any, device="cuda"):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
        self.build_prompt_fn = build_prompt_fn

    def _get_target_loss(self, full_text: str, prompt_text: str) -> float:
        # Токенизируем тексты
        full_tokens = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        prompt_tokens = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        # Безопасно извлекаем input_ids
        full_input_ids = full_tokens["input_ids"]
        prompt_input_ids = prompt_tokens["input_ids"]

        prompt_len = prompt_input_ids.shape[1]
        full_len = full_input_ids.shape[1]

        # Защита от перекрытия, если токенизатор склеил спецтокены
        if prompt_len >= full_len:
            prompt_len = max(0, full_len - 1)

        # Маскируем промпт (-100 игнорируется при расчете CrossEntropyLoss)
        labels = full_input_ids.clone()
        labels[0, :prompt_len] = -100

        with torch.inference_mode():
            outputs = self.model(input_ids=full_input_ids, labels=labels)
            loss = outputs.loss.item()

        # Форсированная очистка памяти от тензоров
        del full_tokens, prompt_tokens, full_input_ids, prompt_input_ids, labels, outputs
        return loss


class PPLScorer(EntropyScorer):
    """
    Метод 3: Перплексия (экспонента от лосса генерации).
    """
    def score(self, examples: dict) -> list[float]:
        scores = []
        batch_size = len(examples['instruction'])

        for i in range(batch_size):
            row = {k: v[i] for k, v in examples.items()}
            prompt_data = self.build_prompt_fn(row, self.tokenizer, train=False)
            full_data = self.build_prompt_fn(row, self.tokenizer, train=True)

            try:
                loss = self._get_target_loss(full_data['text'], prompt_data['text'])
                scores.append(float(np.exp(loss)))
            except Exception:
                print(f"\n[PPL Error] Пример {i}:")
                traceback.print_exc()
                scores.append(np.nan)

        return scores


class IFDScorer(EntropyScorer):
    """
    Метод 4: Instruction-Following Difficulty (IFD).
    Отношение Conditioned Loss к Unconditioned Loss.
    """
    def score(self, examples: dict) -> list[float]:
        scores = []
        batch_size = len(examples['instruction'])

        for i in range(batch_size):
            row = {k: v[i] for k, v in examples.items()}
            output_code = row.get("output", "")

            prompt_data = self.build_prompt_fn(row, self.tokenizer, train=False)
            full_data = self.build_prompt_fn(row, self.tokenizer, train=True)

            try:
                # 1. Лосс генерации кода с учетом инструкции
                cond_loss = self._get_target_loss(full_data['text'], prompt_data['text'])

                # 2. Лосс генерации кода без инструкции (пустой промпт)
                uncond_loss = self._get_target_loss(output_code, "")

                if uncond_loss == 0:
                    scores.append(np.nan)
                else:
                    scores.append(float(cond_loss / uncond_loss))
            except Exception:
                print(f"\n[IFD Error] Пример {i}:")
                traceback.print_exc()
                scores.append(np.nan)

        return scores
