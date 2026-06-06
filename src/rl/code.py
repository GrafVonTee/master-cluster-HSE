from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import signal
from dataclasses import dataclass
from typing import Any, Iterable


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)


@dataclass
class CodeRunResult:
    code: str
    syntax_ok: bool
    passed: int = 0
    total: int = 0
    error: str = ""

    @property
    def pass_ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return float(self.passed) / float(self.total)


def completion_to_text(completion: Any) -> str:
    """Handle TRL/vLLM completion formats: str, dict message, list of messages."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or completion)
    if isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(completion)


def extract_python_code(text: Any) -> str:
    raw = completion_to_text(text)
    raw = _THINK_RE.sub("", raw).strip()
    match = _CODE_BLOCK_RE.search(raw)
    if match:
        return match.group(1).strip()
    idx = raw.find("def ")
    if idx >= 0:
        return raw[idx:].strip()
    idx = raw.find("class ")
    if idx >= 0:
        return raw[idx:].strip()
    return raw.strip()


def normalize_tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
    except Exception:
        pass
    # Accept newline-separated asserts as a pragmatic fallback.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(line.startswith(("assert ", "print(", "pytest")) for line in lines):
        return lines
    return [text]


def is_probably_python(code: str) -> bool:
    return bool(re.search(r"\b(def|class|return|import|for|while|if|elif|else|try|except)\b", code))


def syntax_score(code: str) -> float:
    if not code.strip():
        return -0.5
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return -0.5
    except Exception:
        return 0.0


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal behaviour
    raise TimeoutError("test timeout")


@contextlib.contextmanager
def _time_limit(seconds: float):
    if hasattr(signal, "SIGALRM") and seconds and seconds > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


def run_python_tests(code_text: Any, tests: Iterable[str] | None, timeout: float = 2.0) -> CodeRunResult:
    code = extract_python_code(code_text)
    tests_list = [str(t).strip() for t in (tests or []) if str(t).strip()]
    if not tests_list:
        ok = syntax_score(code) > 0
        return CodeRunResult(code=code, syntax_ok=ok, passed=0, total=0, error="")

    try:
        compiled = compile(code, "<generated>", "exec")
    except Exception as exc:
        return CodeRunResult(code=code, syntax_ok=False, passed=0, total=len(tests_list), error=f"compile: {exc}")

    passed = 0
    errors: list[str] = []
    for test in tests_list:
        ns: dict[str, Any] = {}
        try:
            with _time_limit(timeout):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compiled, ns, ns)
                    exec(test, ns, ns)
            passed += 1
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return CodeRunResult(code=code, syntax_ok=True, passed=passed, total=len(tests_list), error="\n".join(errors[:3]))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out
