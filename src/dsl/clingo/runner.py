from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Any
import contextlib
import io
import os

import clingo


@dataclass
class ClingoSolveResult:
    ok: bool
    satisfiable: bool = False
    unsatisfiable: bool = False
    atoms: list[str] = field(default_factory=list)
    models: list[list[str]] = field(default_factory=list)
    error: str = ""
    timed_out: bool = False


@contextlib.contextmanager
def suppress_native_stderr():
    """
    clingo may write parser errors through native stderr, not Python sys.stderr.
    Since scoring intentionally evaluates broken generated programs, suppress fd=2
    inside the worker process to keep Slurm logs readable.
    """
    old_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)


def _solve_worker(program: str, max_models: int, queue: Queue) -> None:
    try:
        stderr_buf = io.StringIO()

        with contextlib.redirect_stderr(stderr_buf), suppress_native_stderr():
            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], program)
            ctl.ground([("base", [])])

            models: list[list[str]] = []
            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    shown = model.symbols(shown=True)
                    if shown:
                        atoms = sorted(str(s) for s in shown)
                    else:
                        atoms = sorted(str(s) for s in model.symbols(atoms=True))
                    models.append(atoms)
                    if len(models) >= max_models:
                        break
                result = handle.get()

        atom_union = sorted({a for m in models for a in m})
        queue.put(
            {
                "ok": True,
                "satisfiable": bool(result.satisfiable),
                "unsatisfiable": bool(result.unsatisfiable),
                "atoms": atom_union,
                "models": models,
                "error": "",
                "timed_out": False,
            }
        )
    except Exception as e:
        queue.put(
            {
                "ok": False,
                "satisfiable": False,
                "unsatisfiable": False,
                "atoms": [],
                "models": [],
                "error": repr(e),
                "timed_out": False,
            }
        )


def solve_clingo(program: str, timeout: float = 3.0, max_models: int = 1) -> ClingoSolveResult:
    queue: Queue = Queue()
    proc = Process(target=_solve_worker, args=(program, max_models, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        return ClingoSolveResult(ok=False, error="timeout", timed_out=True)

    if queue.empty():
        return ClingoSolveResult(ok=False, error="no_result")

    payload: dict[str, Any] = queue.get()
    return ClingoSolveResult(**payload)
