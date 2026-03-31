import os


def _load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_load_env_file(os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SAMPLE_DATA_DIR = "D:/Data_Store/LV_KHMT/Data/backend/data/samples"
SAMPLE_IMAGES_DIR = "D:/Data_Store/LV_KHMT/Data/backend/data/samples/images"
SAMPLE_TEXTS_DIR = "D:/Data_Store/LV_KHMT/Data/backend/data/samples/texts"
