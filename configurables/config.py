import os
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | \n%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)

# Database
DB_HOST                 = os.getenv("DB_HOST")
DB_PORT                 = int(os.getenv("DB_PORT", "5432"))
DB_NAME                 = os.getenv("DB_NAME")
DB_USER                 = os.getenv("DB_USER", "postgres")
DB_PASSWORD             = os.getenv("DB_PASSWORD")

# VoyageAI
VOYAGE_API_KEY          = os.getenv("VOYAGE_API_KEY")
VOYAGE_EMBEDDING_MODEL  = "voyage-4-lite"
SENTENCE_TRANSFORMERS   = "voyageai/voyage-4-nano"

# DeepSeek
DEEPSEEK_API_KEY        = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL          = "deepseek/deepseek-v4-flash"

# LangFuse
LANGFUSE_PUBLIC_KEY     = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY     = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL       = os.getenv("LANGFUSE_BASE_URL")

# Consts
MAX_ARTICLES            = 600
MAX_ARTICLES_TESTING    = 2
CHUNK_SIZE              = 1024
CHUNK_OVERLAP           = int(0.15 * CHUNK_SIZE)
BATCH_SIZE              = 20000