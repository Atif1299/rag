import os
from dotenv import load_dotenv
from huggingface_hub import login

def setup_environment():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path)

    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN not found in .env")

    os.environ["HUGGINGFACE_API_KEY"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    login(token=token)

FRONTEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
