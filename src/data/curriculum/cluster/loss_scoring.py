# Compatibility wrapper.
# The old implementation mixed PPL and IFD through conditional target-only loss.
# The new PPL split uses plain full-text LM loss from src.data.curriculum.ppl.

from src.data.curriculum.ppl import (  # noqa: F401
    PlainPPLScorer,
    score_plain_ppl_chunk,
)


def score_ppl_ifd_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    return score_plain_ppl_chunk(cfg, task_id=task_id)
