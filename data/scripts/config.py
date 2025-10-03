import httpx
import os
from datetime import datetime

API_BASE = "https://api.chess.com/pub"
USER_AGENT = "navnp04@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}

MAX_CONCURRENCY = 6
MAX_RETRIES = 5
TIMEOUT = httpx.Timeout(30.0)

PGN_PATH = os.path.join("data", "raw", "pgn.jsonl")
FEN_PATH = os.path.join("data", "raw", "fen_unnormalized.jsonl")
NORMALIZED_PATH = os.path.join("data", "normalized", "normalized.parquet")

START_MONTH = datetime(2025, 8, 1)
END_MONTH = datetime(2025, 8, 31)

TOTAL_USERNAMES = 25

