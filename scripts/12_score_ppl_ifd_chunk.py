import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.curriculum.cluster.io import read_yaml, setup_offline
from src.data.curriculum.cluster.loss_scoring import score_ppl_ifd_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/score_pythoncodes_cluster.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    setup_offline(cfg)

    score_ppl_ifd_chunk(cfg, task_id=args.task_id)
    print("DONE")


if __name__ == "__main__":
    main()
