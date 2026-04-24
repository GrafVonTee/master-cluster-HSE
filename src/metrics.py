from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ExecutionResult:
	code: str
	passed_tests: int = 0
	total_tests: int = 0
	logs: str = ""
	entropy: float = 0.0

	@property
	def is_passed(self) -> bool:
		return self.total_tests > 0 and self.total_tests == self.passed_tests

	@property
	def pass_ratio(self) -> float:
		if self.total_tests == 0: return 0.0
		return self.passed_tests / self.total_tests


class BaseCodeMetric(ABC):
	def __init__(self, name: str, generation_config: Dict[str, Any]):
		self.name = name
		self.gen_config = generation_config
		self.n_samples = generation_config.get("num_return_sequences", 1)

	@abstractmethod
	def calculate(self, results: List[List[ExecutionResult]]) -> float:
		pass

	def get_config(self):
		return self.gen_config


class GreedyPass(BaseCodeMetric):
	def __init__(self):
		super().__init__(
			name="greedy@1",
			generation_config={
				"temperature": 0.0,
				"do_sample": False,
				"num_return_sequences": 1,
			}
		)

	def calculate(self, results: List[List[ExecutionResult]]) -> float:
		passed_count = sum(1 for task_res in results if task_res[0].is_passed)
		return passed_count / len(results)


class PassAtk(BaseCodeMetric):
	def __init__(self, k: int, n_samples: int, temperature: float = 0.6):
		super().__init__(
			name=f"pass@{k} (n={n_samples})",
			generation_config={
				"temperature": temperature,
				"do_sample": True,
				"num_return_sequences": n_samples,
			}
		)
		self.k = k

	def calculate(self, results):
		scores = []
		for task_result in results:
			c = sum(1 for res in task_result if res.is_passed)
			n = len(task_result)

			if c == 0:
				scores.append(0.0)
				continue

			score = 1
			if n - c >= self.k:
				score -= np.prod(1.0 - self.k / np.arange(n - c + 1, n + 1))
			scores.append(score)

		return np.mean(scores)


class PercentPassed(BaseCodeMetric):
	def __init__(self, temperature: float = 0.6):
		super().__init__(
			name="mean_%passed",
			generation_config={
				"temperature": temperature,
				"do_sample": True,
				"num_return_sequences": 1,
			}
		)

	def calculate(self, results):
		all_ratios = []

		for task_results in results:
			for res in task_results:
				all_ratios.append(res.pass_ratio)

		return np.mean(all_ratios) if all_ratios else 0.0


class MeanEntropy(BaseCodeMetric):
    def __init__(self, temperature: float = 0.6):
        super().__init__(
			name="mean_entropy",
			generation_config={
				"temperature": temperature,
				"do_sample": True,
				"num_return_sequences": 1
			}
		)

    def calculate(self, results):
        entropies = []
        for task_res in results:
            for res in task_res:
                entropies.append(res.entropy)
        return np.mean(entropies) if entropies else 0.0
