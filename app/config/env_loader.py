import os
from dotenv import load_dotenv

def load_environment():
    """
    Load .env only in local development.
    In production (Kubernetes), environment variables come from ConfigMap/Secrets.
    """
    if os.getenv("ENV", "dev") == "dev":
        load_dotenv()
