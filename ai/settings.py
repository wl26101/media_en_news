# text llm setting
from dataclasses import dataclass

@dataclass
class Settings:
    llm_path: str = "~/project/media_en_news/models/Qwen2.5-3B-Instruct-Q4_K_M/qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_port: int = 8080
    llama_path: str = "/home/wl26/.local/bin/llama"
    tts_path: str = "~/project/media_en_news/ai/tts-backend"
    tts_port: int = 8081
    image_exe: str = "/home/wl26/miniconda3/envs/zimage/bin/"
    image_path: str = "/home/wl26/project/ComfyUI/"
    image_port: int = 8188
    image_json_example_file: str = "~/project/ComfyUI/image_z_image_turbo.json"
    json_text_node: str = "57:27"
    json_seed_node: str = "57:3"
    json_save_node: str = "9"

settings = Settings()
