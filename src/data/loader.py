from typing import Callable, List, Any
from datasets import Dataset
from .types import CodingTask

def load_benchmark(
    dataset: Any,
    mapper_fn: Callable[[Any], CodingTask]
) -> List[CodingTask]:
    tasks = []
    iterable = dataset if not isinstance(dataset, dict) else dataset['test']

    for row in iterable:
        try:
            task = mapper_fn(row)
            tasks.append(task)
        except Exception as e:
            print(f"Error processing task {row.get('task_id', '?')}: {e}")

    return tasks
