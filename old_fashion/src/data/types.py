from dataclasses import dataclass, field
from typing import List

@dataclass
class CodingTask:
    prompt: str
    canonical_solution: str
    tests: List[str]      # Список тестов для assert
    stop_tokens: List[str] = field(default_factory=list)
