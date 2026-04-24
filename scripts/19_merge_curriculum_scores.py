import argparse
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.curriculum.cluster.io import read_yaml, setup_offline
from src.data.curriculum.cluster.merge import merge_curriculum_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/score_pythoncodes_cluster.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    setup_offline(cfg)

    merge_curriculum_scores(cfg)
    print("DONE")


if __name__ == "__main__":
    main()
