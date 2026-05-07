from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LEADS_FILE = BASE_DIR / "leads.csv"

CHROMA_COLLECTION_NAME = "inquirybot_documents"
EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
CHAT_MODEL_CHOICES = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5
SIMILARITY_SCORE_THRESHOLD = 0.35

REFUSAL_MESSAGE = (
    "I could not find enough information in the ingested documents to answer that "
    "confidently. Please upload or ingest a relevant document and try again."
)
