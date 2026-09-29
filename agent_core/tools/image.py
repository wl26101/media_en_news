"""Image tool: one z-Image-Turbo image per sentence, served by ComfyUI.

The queue / poll / download plumbing lives on ai.manage_model.Image; this
module only picks the workflow inputs and the destination path, and turns any
failure into an exception the graph node can record.
"""

from __future__ import annotations

import logging
from pyexpat import model
import random
from pathlib import Path

import ai.manage_model as manage_model
from ai.manage_model import Image
from ai.settings import settings

from .. import config
from ..tracing import traced

logger = logging.getLogger(__name__)


def _client() -> Image:
    return Image(image_port=settings.image_port)


@traced
def generate_image(
    prompt: str,
    out_path: str | Path,
    seed: int | None = None,
    timeout: int = config.IMAGE_TIMEOUT,
    model_manage: manage_model.Manage_Model | None = None,
) -> str:
    """Generate one image from `prompt` and save it to `out_path`.

    Returns the path; raises when ComfyUI rejects, fails or times out.
    """
    if model_manage is not None:
        model_manage.launch_model("image")
    client = _client()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    seed = random.randint(0, 2**63 - 1) if seed is None else seed

    workflow = client.inject_prompt_and_seed(prompt, seed, filename_prefix=dest.stem)
    if workflow is None:
        raise RuntimeError("could not build the z-Image-Turbo workflow")
    prompt_id = client.queue_prompt(workflow)
    if prompt_id is None:
        raise RuntimeError("ComfyUI rejected the prompt")
    images = client.wait_for_completion(prompt_id, timeout=timeout)
    if not images:
        raise RuntimeError("ComfyUI produced no image")

    logger.debug("image %s from seed %s", dest.name, seed)
    return client.fetch_image(images[0], str(dest))
