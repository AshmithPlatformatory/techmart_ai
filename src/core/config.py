import os
import logging
from dotenv import load_dotenv
from langfuse import Langfuse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

try:
    langfuse_client = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST")
    )
    logger.info("Langfuse client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Langfuse client: {e}")
    langfuse_client = None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

EXOTEL_API_KEY = os.environ.get("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.environ.get("EXOTEL_API_TOKEN")

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")