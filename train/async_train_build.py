"""Async build-agent training entry (sync pipeline; shares MaskablePPO + vec env stack)."""

import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_build import BuildPipelineManager


def main():
    logger.info("Build async entry: running BuildPipelineManager (sync MaskablePPO path).")
    BuildPipelineManager().run()


if __name__ == "__main__":
    main()
