import multiprocessing as mp
import signal
import sys
import re
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from src.metrics import ExecutionResult
import src.config as config


def extract_code_from_completion(text: str) -> str:
    """Вырезает код из блока ```python ... ``` или возвращает как есть."""
    text = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "def " in text:
        return text[text.find("def "):]
    return text


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout reached")


def _process_single_sample(args: Tuple[str, List[str], float]) -> ExecutionResult:
    """
    Воркер, который выполняется в отдельном процессе пула.
    Принимает (generated_text, tests, timeout).
    """
    generated_text, tests, timeout = args
    clean_code = extract_code_from_completion(generated_text)
    passed_count = 0
    total_count = len(tests)
    logs = ""

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)

    try:
        compiled_code = compile(clean_code, "<string>", "exec")
    except Exception as e:
        return ExecutionResult(clean_code, 0, total_count, logs=f"Syntax Error: {e}")

    for test_case in tests:
        ns = {}

        try:
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, timeout)

            exec(compiled_code, ns, ns)
            exec(test_case, ns, ns)

            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0) # Отключаем таймер

            passed_count += 1

        except TimeoutException:
            logs += f"Test timed out.\n"
        except Exception as e:
            pass
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)

    return ExecutionResult(
        code=clean_code,
        passed_tests=passed_count,
        total_tests=total_count,
        logs=logs
    )


class LocalExecutor:
    def __init__(self, max_workers: int = None):
        default_workers = getattr(config, "NUM_PROCESSES", mp.cpu_count())
        self.max_workers = max_workers if max_workers else default_workers

    def batch_execute(self,
                      tasks: List[Tuple[str, List[str]]],
                      timeout_per_test: float = 2.0) -> List[ExecutionResult]:
        """
        Параллельный запуск.
        """
        # Готовим аргументы
        map_args = [(text, tests, timeout_per_test) for text, tests in tasks]

        print(f"🚀 Executing {len(tasks)} samples in parallel using {self.max_workers} workers...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_single_sample, map_args),
                total=len(map_args),
                desc="Running Tests"
            ))

        return results

    def execute(self, generated_text: str, tests: List[str]) -> ExecutionResult:
        """Совместимость для одиночного запуска"""
        return _process_single_sample((generated_text, tests, 2.0))
