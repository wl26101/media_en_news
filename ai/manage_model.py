from pydantic import BaseModel
import json
import logging
import os
import random
import subprocess
from .settings import settings
import requests
import time
from requests import HTTPError

logger = logging.getLogger(__name__)

class LlamaServe:
    def __init__(self, model_path: str = settings.llm_path, port: int = settings.llm_port):
        self.model_path = os.path.expanduser(model_path)
        self.port = port
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([
            settings.llama_path, "serve", "-m", self.model_path,
            "--port", str(self.port), "--host", "0.0.0.0"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def is_health(self, timeout=2):
        """检测http服务是否真正就绪"""
        if not (self.proc and self.proc.poll() is None):
            return False
        try:
            url = f"http://127.0.0.1:{self.port}/health"
            resp = requests.get(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False
    def wait_ready(self, max_wait=60):
            """阻塞等待模型加载完成，成功返回True，超时False"""
            for _ in range(max_wait):
                if self.is_health():
                    logger.info("llm 服务就绪")
                    return True
                time.sleep(1)
            logger.error("llm启动超时，未就绪")
            return False
    def send_llm_prompt(self, prompt: str):
        if self.proc and self.proc.poll() is None:
            try:
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300,
                }
                response = requests.post(
                    f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                resp_json = response.json()
                content = resp_json["choices"][0]["message"]["content"]
                return content
            except Exception as e:
                logger.error(f"Failed to send prompt: {e}")
        else:
            logger.warning("Cannot send prompt, the LlamaServe process is not running.")
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TTS():
    def __init__(self, tts_path: str = settings.tts_path, tts_port: int = settings.tts_port):
        self.tts_path = tts_path
        self.tts_port = tts_port
        self.proc = None
        self.uvicorn_bin = os.path.expanduser("~/miniconda3/envs/tts/bin/uvicorn")
    
    def start(self):
        # run fastapi dev in ~/project/media_en_news/ai/tts-backend, use conda tts env
        tts_backend_path = os.path.expanduser(self.tts_path)
        # use conda run uvicorn app.main:app --host 0.0.0.0 --port 8000 
        self.proc = subprocess.Popen(
            [self.uvicorn_bin, "app.main:app", "--host", "0.0.0.0", "--port", str(self.tts_port)],
            cwd=tts_backend_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        return self
    
    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def is_health(self, timeout=2):
        """检测http服务是否真正就绪"""
        if not (self.proc and self.proc.poll() is None):
            return False
        try:
            url = f"http://127.0.0.1:{self.tts_port}/health"
            resp = requests.get(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def wait_ready(self, max_wait=60):
        """阻塞等待模型加载完成，成功返回True，超时False"""
        for _ in range(max_wait):
            if self.is_health():
                logger.info("TTS 服务就绪")
                return True
            time.sleep(1)
        logger.error("TTS启动超时，未就绪")
        return False

    def get_health(self, timeout=5):
        if not (self.proc and self.proc.poll() is None):
            return None
        try:
            url = f"http://127.0.0.1:{self.tts_port}/health"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to query TTS health: {e}")
            return None
    
    def generate(self, text: str, language: str = "English", tone=None, output_path: str = "out.wav"):
        if not (self.proc and self.proc.poll() is None):
            logger.warning("TTS service is not running.")
            return None
        try:
            url = f"http://127.0.0.1:{self.tts_port}/v1/tts"
            payload = {
                "text": text,
                "language": language,
            }
            tone_value = tone.strip() if isinstance(tone, str) else tone
            if tone_value:
                health = self.get_health()
                if health and not health.get("tone_supported", False):
                    model_type = health.get("model_type", "unknown")
                    logger.warning(
                        "TTS backend model '%s' does not support tone control; ignoring tone=%r.",
                        model_type,
                        tone_value,
                    )
                else:
                    payload["tone"] = tone_value

            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return output_path

            return resp.content
        except HTTPError as e:
            detail = None
            response = e.response
            if response is not None:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text.strip() or None
            if detail is not None:
                logger.error(f"Failed to generate TTS: {e}; detail={detail}")
            else:
                logger.error(f"Failed to generate TTS: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate TTS: {e}")
            return None

class Image():
    """z-Image-Turbo image generation served by a local ComfyUI instance.

    The workflow JSON in settings.image_json_example_file is injected with the
    prompt/seed/filename_prefix, queued on ComfyUI, awaited, and the resulting
    image is downloaded to the requested path.
    """

    QUEUE_TIMEOUT = 60    # seconds to submit the prompt
    RUN_TIMEOUT = 600     # seconds to wait for one image to finish
    POLL_INTERVAL = 2.0

    def __init__(self, image_exe: str = settings.image_exe, image_path: str = settings.image_path, image_port: int = settings.image_port):
        self.image_exe = image_exe
        self.image_path = image_path
        self.image_port = image_port
        self.proc = None
    def start(self):
        self.proc = subprocess.Popen(
            [self.image_exe + "python", "main.py", "--port", str(self.image_port)],
            cwd=self.image_path,
        )
    def stop(self):
        if hasattr(self, "proc") and self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception as e:
                logger.error(f"Failed to stop image process: {e}")
    def get_health(self, timeout=5):
        if not (self.proc and self.proc.poll() is None):
            return None
        try:
            url = f"http://127.0.0.1:{self.image_port}/system_stats"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to query image health: {e}")
            return None
    def wait_ready(self, timeout=30):
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            health = self.get_health()
            if health is not None:
                return True
            time.sleep(1)
        return False
    def inject_prompt_and_seed(self, prompt: str, seed: int, filename_prefix: str = None) -> dict:
        """Load the z-Image-Turbo workflow and inject the prompt, seed and
        (optional) SaveImage filename prefix."""
        try:
            workflow_file = os.path.expanduser(settings.image_json_example_file)
            with open(workflow_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data[settings.json_text_node]["inputs"]["text"] = prompt
            data[settings.json_seed_node]["inputs"]["seed"] = seed
            save_node = getattr(settings, "json_save_node", None)
            if filename_prefix and save_node and save_node in data:
                data[save_node]["inputs"]["filename_prefix"] = filename_prefix
            return data
        except Exception as e:
            logger.error(f"Failed to inject prompt and seed: {e}")
            return None

    def queue_prompt(self, workflow: dict, timeout: int = QUEUE_TIMEOUT) -> str:
        """Queue a workflow on ComfyUI; returns the prompt_id."""
        try:
            url = f"http://127.0.0.1:{self.image_port}/prompt"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"prompt": workflow},
                timeout=timeout,
            )
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                logger.error(f"ComfyUI did not accept the prompt: {resp.text}")
                return None
            return prompt_id
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None

    def wait_for_completion(self, prompt_id: str, timeout: int = RUN_TIMEOUT) -> list:
        """Poll /history/{prompt_id} until the run finishes; returns the image
        records from the SaveImage node."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{self.image_port}/history/{prompt_id}",
                    timeout=10,
                )
                resp.raise_for_status()
                history = resp.json()
            except Exception as e:
                logger.warning(f"Failed to query image history: {e}")
                time.sleep(self.POLL_INTERVAL)
                continue

            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    logger.error(f"ComfyUI execution error: {status.get('messages', [])}")
                    return None
                images = []
                for node_output in entry.get("outputs", {}).values():
                    images.extend(node_output.get("images", []))
                if images or status.get("completed"):
                    return images
            time.sleep(self.POLL_INTERVAL)
        logger.error(f"ComfyUI prompt {prompt_id} did not finish within {timeout}s")
        return None

    def fetch_image(self, image: dict, dest: str = "output.png") -> str:
        """Download one generated image from /view into dest."""
        resp = requests.get(
            f"http://127.0.0.1:{self.image_port}/view",
            params={
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            },
            timeout=60
        )
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return dest

    def generate(self, prompt: str, output_path: str = "output.png", seed: int = None, timeout: int = RUN_TIMEOUT) -> str:
        """Generate a single image from prompt and save it to output_path.

        Returns output_path on success, None on failure.
        """
        if not (self.proc and self.proc.poll() is None):
            logger.warning("Image service is not running.")
            return None
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        try:
            dest = os.path.expanduser(output_path)
            parent = os.path.dirname(os.path.abspath(dest))
            os.makedirs(parent, exist_ok=True)
            # prefix the ComfyUI-side save file after the destination stem so the
            # generated file is identifiable in the ComfyUI output dir
            stem = os.path.splitext(os.path.basename(dest))[0]

            workflow = self.inject_prompt_and_seed(prompt, seed, filename_prefix=stem)
            if workflow is None:
                return None
            prompt_id = self.queue_prompt(workflow)
            if prompt_id is None:
                return None
            images = self.wait_for_completion(prompt_id, timeout=timeout)
            # log images info
            logger.info(f"Generated images info: {images}")
            if not images:
                logger.error("ComfyUI workflow finished but produced no images.")
                return None
            self.fetch_image(images[0], dest)
            logger.info(f"Image saved to {dest} (seed={seed})")
            return dest
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return None

class Manage_Model():

    def __init__(self):
        self.status = { "llm": 0, "tts": 0, "image": 0 }
        self.llm_serve = LlamaServe()
        self.tts_serve = TTS()
        self.image_serve = Image()

    def get_status(self):
        return self.status

    def close_model(self, model_name: str, provider: str = "local"):
        if provider != "local":
            logger.info("No need to close the model when the provider is not local.")
            return
        else:
            logger.info(f"Closing the model {model_name} with provider {provider}.")
            if model_name == "llm":
                # check llm status
                if self.status[model_name] == 0:
                    logger.info(f"LLM model is not running.")
                else:
                    self.llm_serve.stop()
                    self.status[model_name] = 0
            elif model_name == "tts":
                # check tts status
                if self.status[model_name] == 0:
                    logger.info(f"TTS model is not running.")
                else:
                    self.tts_serve.stop()
                    self.status[model_name] = 0
            elif model_name == "image":
                # check image status
                if self.status[model_name] == 0:
                    logger.info(f"Image model is not running.")
                else:
                    self.image_serve.stop()
                    self.status[model_name] = 0
            else:
                logger.warning(f"Unknown model name: {model_name}")

    def launch_model(self, model_name: str, provider: str = "local"):
        if provider != "local":
            logger.info("No need to launch the model when the provider is not local.")
            return
        else:
            logger.info(f"Launching the model {model_name} with provider {provider}.")
            if model_name == "llm":
                # check llm status
                if self.status[model_name] == 1:
                    logger.info(f"LLM model is already running.")
                    return
                else:
                    # check image status
                    if self.status["image"] == 1:
                        logger.info("Image model is running. Launching LLM may affect performance.Close image serve.")
                        self.image_serve.stop()
                        self.status["image"] = 0
                    self.llm_serve.start()
                    self.llm_serve.wait_ready()
                    self.status[model_name] = 1
            elif model_name == "tts":
                # check tts status
                if self.status[model_name] == 1:
                    logger.info(f"TTS model is already running.")
                    return
                else:
                    # check image status
                    if self.status["image"] == 1:
                        logger.info("Image model is running. Launching TTS may affect performance.Close image serve.")
                        self.image_serve.stop()
                        self.status["image"] = 0
                    self.tts_serve.start()
                    self.tts_serve.wait_ready()
                    self.status[model_name] = 1
            elif model_name == "image":
                # check image status
                if self.status[model_name] == 1:
                    logger.info(f"Image model is already running.")
                    return
                else:
                    # check llm status
                    if self.status["llm"] == 1:
                        logger.info("LLM model is running. Launching Image may affect performance.Close llm serve.")
                        self.llm_serve.stop()
                        self.status["llm"] = 0
                    # check tts status
                    if self.status["tts"] == 1:
                        logger.info("TTS model is running. Launching Image may affect performance.Close tts serve.")
                        self.tts_serve.stop()
                        self.status["tts"] = 0
                    self.image_serve.start()
                    self.image_serve.wait_ready()
                    self.status[model_name] = 1
            else:
                logger.warning(f"Unknown model name: {model_name}")




