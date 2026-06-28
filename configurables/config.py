import os
from dotenv import load_dotenv

load_dotenv()

# Database
DB_HOST                 = os.getenv("DB_HOST")
DB_PORT                 = int(os.getenv("DB_PORT", "5432"))
DB_NAME                 = os.getenv("DB_NAME", "threatscope")
DB_USER                 = os.getenv("DB_USER", "postgres")
DB_PASSWORD             = os.getenv("DB_PASSWORD")

# VoyageAI
VOYAGE_API_KEY          = os.getenv("VOYAGE_API_KEY")

# DeepSeek
DEEPSEEK_API_KEY        = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL          = "deepseek-v4-flash"

# LangFuse
LANGFUSE_PUBLIC_KEY     = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY     = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL       = os.getenv("LANGFUSE_BASE_URL")