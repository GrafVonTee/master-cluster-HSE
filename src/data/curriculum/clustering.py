import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Any
from .base import BaseScorer

class LexicalClusterScorer(BaseScorer):
    """
    Метод 5: Лексическая кластеризация.
    Использует TF-IDF поверх токенизатора LLM и ищет примеры с редким словарем.
    """
    def __init__(self, name: str, tokenizer: Any, n_clusters: int = 10):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Настраиваем TF-IDF так, чтобы он использовал наш токенизатор
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: tokenizer.encode(x, add_special_tokens=False),
            token_pattern=None,
            lowercase=False
        )

    def fit(self, dataset):
        print(f"[{self.name}] Обучение TF-IDF и KMeans...")
        texts = dataset['output']
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.kmeans.fit(tfidf_matrix)

    def score(self, examples: dict) -> list[float]:
        texts = examples.get('output', [])
        if not texts:
            return []

        tfidf_matrix = self.vectorizer.transform(texts)
        # Считаем расстояние от каждого примера до всех центров
        distances = self.kmeans.transform(tfidf_matrix)
        # Берем расстояние до БЛИЖАЙШЕГО центра (чем больше, тем аномальнее пример)
        min_distances = np.min(distances, axis=1)

        return min_distances.tolist()


class SemanticClusterScorer(BaseScorer):
    """
    Метод 6: Семантическая кластеризация.
    Вытаскивает Hidden States из середины сети для понимания "смысла" кода.
    """
    def __init__(self, name: str, model: Any, tokenizer: Any, device="cuda", n_clusters: int = 10, layer_idx: int = None):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_clusters = n_clusters

        # Включаем выдачу скрытых состояний, если она выключена
        self.model.config.output_hidden_states = True

        # Берем средний слой (где формируются абстракции), если не указан явно
        self.layer_idx = layer_idx if layer_idx is not None else (model.config.num_hidden_layers // 2)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            # Извлекаем тензор нужного слоя: (batch_size, seq_len, hidden_size)
            hidden = outputs.hidden_states[self.layer_idx]

            # Берем вектор последнего токена для каждого примера в батче
            last_token_vecs = hidden[:, -1, :]
            embeddings = last_token_vecs.cpu().float().numpy()

        del inputs, outputs, hidden, last_token_vecs
        return embeddings

    def fit(self, dataset):
        print(f"[{self.name}] Извлечение эмбеддингов для обучения кластеров...")
        # Если датасет гигантский, для фита лучше брать срез (например, 5000)
        texts = dataset['output']

        # Разбиваем на микро-батчи, чтобы не получить OOM при фите
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            emb = self._get_embeddings(batch_texts)
            all_embeddings.append(emb)

        final_embeddings = np.vstack(all_embeddings)
        self.kmeans.fit(final_embeddings)

    def score(self, examples: dict) -> list[float]:
        texts = examples.get('output', [])
        if not texts:
            return []

        embeddings = self._get_embeddings(texts)
        distances = self.kmeans.transform(embeddings)
        min_distances = np.min(distances, axis=1)

        return min_distances.tolist()
