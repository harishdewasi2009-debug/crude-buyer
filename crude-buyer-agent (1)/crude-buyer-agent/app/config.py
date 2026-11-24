import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.getenv("CBA_DATA_DIR", str(BASE_DIR / "data")))
UA_TOKEN = "CrudeBuyerResearchBot"
USER_AGENT = os.getenv("CBA_USER_AGENT", f"{UA_TOKEN}/1.0 (B2B market research; respects robots.txt)")
