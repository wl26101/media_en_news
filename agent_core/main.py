#!/usr/bin/env python3
"""Run the agentic news-video pipeline once.

    python -m agent_core.main                     # full run from resource/text/source.txt
    python agent_core/main.py --no-launch         # use model servers that are already up

The run keeps every intermediate artifact under resource/ (see config.py) and
writes resource/video/news_manifest.json describing the result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Running this file directly (`python main.py`) puts agent_core/ on sys.path
# instead of the project root, so the package and the shared utils/ai modules
# would not be importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import config
from agent_core.graph import build_graph
from agent_core.graph import model_serve
from agent_core.tools import models, text as text_tools
from agent_core.tracing import setup_tracing
from utils.logger_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an English news video from resource/text/source.txt.")
    parser.add_argument("--source", default=str(config.SOURCE_FILE), help="news text to summarize")
    parser.add_argument("--no-launch", action="store_true", help="assume the llm/tts/image servers are already running")
    parser.add_argument("--keep-models", action="store_true", help="do not stop the model servers when done")
    parser.add_argument("--keep-files", action="store_true", help="keep the artifacts of the previous run")
    parser.add_argument("--fps", type=int, default=config.VIDEO_FPS, help="frames per second for the still images")
    parser.add_argument("--recursion-limit", type=int, default=25, help="langgraph step budget")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    traced = setup_tracing()

    try:
        if not args.keep_files:
            removed = text_tools.clean_outputs()
            logger.info("cleaned %d artifact(s) from the previous run", len(removed))

        logger.info("running the news-video workflow on %s", args.source)
        graph = build_graph()
        result = graph.invoke(
            {"source_file": args.source, "attempts": 0, "errors": [], "log": []},
            config={
                "recursion_limit": args.recursion_limit,
                "run_name": "news-video",
                "metadata": {"source": args.source, "stem": config.STEM, "langsmith_tracing": traced},
            },
        )
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception:
        logger.exception("the workflow failed")
        return 1
    finally:
        if not args.keep_models:
            status = model_serve.get_status()
            logger.info("model server status: %s", status)
            if status.get("llm") == 1:
                logger.info("stopping the model server")
                model_serve.close_model("llm")
            if status.get("tts") == 1:
                logger.info("stopping the model server")
                model_serve.close_model("tts")
            if status.get("image") == 1:
                logger.info("stopping the model server")
                model_serve.close_model("image")

    for line in result.get("log", []):
        logger.info("%s", line)
    if not result.get("video_file"):
        logger.error("no video was produced; see the errors above")
        return 1
    logger.info("video: %s", result["video_file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
