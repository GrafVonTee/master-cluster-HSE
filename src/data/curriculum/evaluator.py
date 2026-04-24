import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

class CurriculumEvaluator:
    """
    Класс для визуального и статистического анализа размеченного датасета.
    Генерирует графики для курсовой работы/статьи.
    """
    def __init__(self, processed_dataset: Dataset, save_dir: str = "./plots"):
        # Переводим датасет в pandas DataFrame для удобства аналитики
        print("Конвертация датасета в DataFrame для анализа...")
        self.df = processed_dataset.to_pandas()
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Находим все колонки с оценками и категориями
        self.score_cols = [c for c in self.df.columns if c.endswith('_score')]
        self.cat_cols = [c for c in self.df.columns if c.endswith('_category')]

        # Настраиваем стиль графиков (академический, чистый)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

    def plot_distributions(self):
        """
        Строит гистограммы с кривой плотности (KDE) для каждой метрики.
        Позволяет оценить разброс значений.
        """
        print("Генерация графиков распределения (Distributions)...")
        n_cols = len(self.score_cols)
        fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 5 * ((n_cols + 1) // 2)))
        axes = axes.flatten()

        for i, col in enumerate(self.score_cols):
            ax = axes[i]
            # Отбрасываем NaN для корректной отрисовки
            data = self.df[col].dropna()

            # Для метрик с сильным правым хвостом (например, длина или PPL)
            # иногда полезно брать логарифм, но для чистоты эксперимента покажем как есть
            sns.histplot(data, kde=True, ax=ax, bins=50, color='royalblue')

            # Считаем перцентили для отображения вертикальных линий среза
            if len(data) > 0:
                p33, p66 = np.percentile(data, [33, 66])
                ax.axvline(p33, color='forestgreen', linestyle='--', label=f'33% (Easy: <= {p33:.2f})')
                ax.axvline(p66, color='firebrick', linestyle='--', label=f'66% (Hard: > {p66:.2f})')

            ax.set_title(f'Распределение метрики: {col.replace("_score", "")}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Количество примеров')
            ax.legend()

        # Удаляем пустые сабплоты, если метрик нечетное количество
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "score_distributions.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Сохранено: {save_path}")

    def plot_correlations(self):
        """
        Строит тепловую карту ранговой корреляции Спирмена.
        Ранговая корреляция устойчива к выбросам и нелинейностям.
        """
        print("Генерация матрицы корреляций (Spearman)...")
        # Берем только колонки со скорами и убираем суффикс '_score' для красоты
        scores_df = self.df[self.score_cols].rename(columns=lambda x: x.replace('_score', ''))

        # Считаем корреляцию Спирмена (игнорирует NaN автоматически)
        corr_matrix = scores_df.corr(method='spearman')

        plt.figure(figsize=(10, 8))
        # Используем дивергентную палитру: синий - отрицательная, красный - положительная корреляция
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )

        plt.title('Матрица корреляций Спирмена между метриками сложности', pad=20)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "correlation_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Сохранено: {save_path}")

    def plot_categories(self):
        """
        Строит столбчатые диаграммы баланса классов (Easy/Medium/Hard).
        """
        print("Генерация графиков баланса классов...")
        n_cols = len(self.cat_cols)
        fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(14, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten()

        # Задаем фиксированные цвета для консистентности
        color_map = {'easy': 'forestgreen', 'medium': 'gold', 'hard': 'firebrick'}

        for i, col in enumerate(self.cat_cols):
            ax = axes[i]

            # Подсчет уникальных значений
            counts = self.df[col].value_counts().reindex(['easy', 'medium', 'hard']).fillna(0)

            bars = ax.bar(counts.index, counts.values, color=[color_map.get(x, 'gray') for x in counts.index])

            ax.set_title(f'Баланс категорий: {col.replace("_category", "")}')
            ax.set_ylabel('Количество примеров')

            # Добавляем текстовые подписи над столбцами
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "category_balance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Сохранено: {save_path}")

    def generate_report(self):
        """
        Запускает генерацию всех графиков и выводит краткую текстовую статистику.
        """
        print(f"=== Запуск оценки датасета (Total samples: {len(self.df)}) ===")
        self.plot_distributions()
        self.plot_correlations()
        self.plot_categories()
        print("=== Оценка завершена. Графики сохранены в директорию. ===")
