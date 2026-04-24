import numpy as np
from abc import ABC, abstractmethod
from datasets import Dataset
import uuid

class BaseScorer(ABC):
	def __init__(self, name: str):
		self.name = name

	def fit(self, dataset: Dataset):
		# опциональный метод
		pass

	@abstractmethod
	def score(self, batch: dict) -> list[float]:
		pass


class CurriculumPipeline:
	def __init__(self, scorers: list[BaseScorer], percentiles: list[float] = [.33, .66]):
		self.scorers = scorers
		self.percentiles = percentiles
		self.fitted = False

	def fit(self, dataset: Dataset):
		for scorer in self.scorers:
			if scorer.fit is not BaseScorer.fit:
				print(f"[{scorer.name}] Обучение...")
			scorer.fit(dataset)
		self.fitted = True

	def process_dataset(self, dataset: Dataset, batch_size: int = 100) -> Dataset:
		if not self.fitted:
			self.fit(dataset)

		for scorer in self.scorers:
			print(f"[{scorer.name}] Оцениваем датасет по степеням...")

			dataset = dataset.map(
				lambda batch: {f"{scorer.name}_score": scorer.score(batch)},
				batched = True,
				batch_size = batch_size,
				desc = f"Scoring {scorer.name}",
				new_fingerprint=f"{scorer.name}_{uuid.uuid4().hex}",
			)

		return self._assign_categories(dataset)

	def _assign_categories(self, dataset: Dataset):
		score_columns = [col for col in dataset.column_names if col.endswith("_score")]
		thresholds = {}

		print("Расчёт перцентилей для присвоения категорий")
		for col in score_columns:
			scores = np.array(dataset[col])
			valid_scores = scores[~np.isnan(scores)]

			if len(valid_scores) > 0:
				p_low, p_high = np.percentile(valid_scores, list(self.percentiles))
				thresholds[col] = (p_low, p_high)
			else:
				thresholds[col] = (0, 0)

		def categorize(row):
			for col in score_columns:
				base_name = col.replace("_score", "")
				score = row[col]

				if np.isnan(score):
					row[f"{base_name}_category"] = "medium"
					continue

				p_low, p_high = thresholds[col]

				if score <= p_low:
					row[f"{base_name}_category"] = "easy"
				elif score <= p_high:
					row[f"{base_name}_category"] = "medium"
				else:
					row[f"{base_name}_category"] = "hard"

			return row

		return dataset.map(categorize, desc="Categorizing")
