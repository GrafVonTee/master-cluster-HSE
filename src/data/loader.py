from typing import Callable, List, Any
from datasets import Dataset
from .types import CodingTask

def load_benchmark(
    dataset: Any,
    mapper_fn: Callable[[Any], CodingTask]
) -> List[CodingTask]:
    tasks = []
    iterable = dataset if not isinstance(dataset, dict) else dataset['test']

    failed = 0
    for row in iterable:
        try:
            task = mapper_fn(row)
            tasks.append(task)
        except Exception as e:
            failed += 1
            print(f"Error processing task {row.get('task_id', '?')}: {e}")

    if not tasks:
        raise RuntimeError(
            f"load_benchmark produced 0 tasks from {len(iterable)} rows; "
            f"failed={failed}. Check benchmark mapper and dataset columns."
        )

    return tasks
