import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TODAY"] = "2026-11-23"
for k in ("TAVILY_API_KEY", "LLM_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
    os.environ.pop(k, None)
