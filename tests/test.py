import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.logger_config import setup_logging
from ai.manage_model import Manage_Model

setup_logging()

import logging
logger = logging.getLogger(__name__)
logger.info("Logger is set up successfully.")
logger.debug("This is a debug message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")

def test_llm():
    manage_model = Manage_Model()
    logger.info(f"Initial model status: {manage_model.get_status()}")
    assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}

    try:
        manage_model.launch_model("llm")
        logger.info(f"Model status after launching LLM: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 1, "tts": 0, "image": 0}

        # send prompt to llm serve
        res = manage_model.llm_serve.send_llm_prompt("Introduce apple.")
        logger.info(f"LLM result: {res}")
        assert res is not None
    finally:
        manage_model.close_model("llm")
        logger.info(f"Model status after closing LLM: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}


def test_tts():
    manage_model = Manage_Model()
    logger.info(f"Initial model status: {manage_model.get_status()}")
    assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}

    try:
        manage_model.launch_model("tts")
        logger.info(f"Model status after launching TTS: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 1, "image": 0}

        res = manage_model.tts_serve.generate("Hello world.", language="English", tone="cheerful", output_path="out2.wav")
        logger.info(f"TTS result saved to out.wav: {res}")
        assert res is not None
    finally:
        manage_model.close_model("tts")
        logger.info(f"Model status after closing TTS: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}
        
def test_image():
    manage_model = Manage_Model()
    logger.info(f"Initial model status: {manage_model.get_status()}")
    assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}

    try:
        manage_model.launch_model("image")
        logger.info(f"Model status after launching Image: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 1}

        # send request to image serve
        res = manage_model.image_serve.get_health()
        logger.info(f"Image health: {res}")
        assert res is not None

        # generate an image "Harry Potter played football"
        res = manage_model.image_serve.generate("Harry Potter played football", output_path="out.png")
        logger.info(f"Generated image saved to out.png: {res}")
        assert res is not None
    finally:
        manage_model.close_model("image")
        logger.info(f"Model status after closing Image: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}

# test multiple models running simultaneously
def test_multiple_models():
    manage_model = Manage_Model()
    logger.info(f"Initial model status: {manage_model.get_status()}")
    assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}

    try:
        manage_model.launch_model("llm")
        manage_model.launch_model("tts")
        logger.info(f"Model status after launching LLM and TTS: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 1, "tts": 1, "image": 0}

        manage_model.launch_model("image")
        logger.info(f"Model status after launching Image: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 1}

        manage_model.launch_model("tts")
        logger.info(f"Model status after launching TTS again: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 1, "image": 0}
    finally:
        manage_model.close_model("llm")
        manage_model.close_model("tts")
        manage_model.close_model("image")
        logger.info(f"Model status after closing LLM, TTS, and Image: {manage_model.get_status()}")
        assert manage_model.get_status() == {"llm": 0, "tts": 0, "image": 0}
