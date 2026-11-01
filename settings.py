"""Machine-specific paths and addresses. Copy settings.example.json to settings.json and edit it, or set the environment variables
(they win over the file). Everything the scripts write goes to WORK."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
_p = os.path.join(HERE, "settings.json")
_cfg = json.load(open(_p, encoding="utf-8")) if os.path.exists(_p) else {}


def _get(key, env, default=None):
    return os.environ.get(env) or _cfg.get(key) or default


COMFY_URL = _get("comfy_url", "SPRITE_FORGE_COMFY_URL", "http://127.0.0.1:8188")
COMFY_INPUT = _get("comfy_input", "SPRITE_FORGE_COMFY_INPUT")        # ComfyUI/input: images are copied here for LoadImage
COMFY_OUTPUT = _get("comfy_output", "SPRITE_FORGE_COMFY_OUTPUT")      # ComfyUI/output: where SaveImage writes
WORK = _get("work_dir", "SPRITE_FORGE_WORK", os.path.join(HERE, "work"))
BASE_BODIES = _get("base_bodies", "SPRITE_FORGE_BASE_BODIES", os.path.join(HERE, "base_bodies"))
NODE22 = _get("node22", "SPRITE_FORGE_NODE22", "node")                # Node >= 22.18, for pxl
PXL = _get("pxl", "SPRITE_FORGE_PXL", os.path.join(os.path.expanduser("~"), ".claude", "skills", "pxl-pixel-art", "bin", "pxl.js"))
os.makedirs(WORK, exist_ok=True)


def need_comfy():
    if not (COMFY_INPUT and COMFY_OUTPUT):
        raise SystemExit("Set comfy_input and comfy_output in settings.json (see settings.example.json).")
