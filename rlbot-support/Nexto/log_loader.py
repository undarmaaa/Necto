import os
import re
import json
from datetime import datetime, date
from typing import Generator, Iterable, List, Optional, Tuple

# Directory where date-sharded logs live
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

RAW_FILE_REGEX = re.compile(r"^nexto_clears_(\d{4}-\d{2}-\d{2})\.jsonl$")
LABELED_FILE_REGEX = re.compile(r"^nexto_clears_labeled_(\d{4}-\d{2}-\d{2})\.jsonl$")
DEDUPED_FILE_REGEX = re.compile(r"^nexto_clears_deduped_(\d{4}-\d{2}-\d{2})\.jsonl$")

def ensure_logs_dir() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    return LOGS_DIR

def _parse_date_str(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()

def _date_range_filter(d: date, start: Optional[date], end: Optional[date]) -> bool:
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True

def _collect_shards(labeled: bool) -> List[Tuple[date, str]]:
    """
    Returns list of (date, full_path) pairs for either raw or labeled shards.
    """
    if not os.path.isdir(LOGS_DIR):
        return []
    entries = []
    for name in os.listdir(LOGS_DIR):
        m = (LABELED_FILE_REGEX if labeled else RAW_FILE_REGEX).match(name)
        if not m:
            continue
        d = _parse_date_str(m.group(1))
        entries.append((d, os.path.join(LOGS_DIR, name)))
    # Sort by date ascending
    entries.sort(key=lambda x: x[0])
    return entries

def iter_shard_paths(start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     labeled: bool = False) -> Generator[str, None, None]:
    """
    Yield full paths to shards within an optional inclusive date range.
    Dates are "YYYY-MM-DD" in UTC.
    """
    start_d = _parse_date_str(start_date) if start_date else None
    end_d = _parse_date_str(end_date) if end_date else None
    for d, path in _collect_shards(labeled=labeled):
        if _date_range_filter(d, start_d, end_d):
            yield path

def resolve_latest(labeled: bool = False) -> Optional[str]:
    """
    Return the latest shard path if available, else None.
    """
    shards = _collect_shards(labeled=labeled)
    if not shards:
        return None
    return shards[-1][1]

def shard_date_from_path(path: str, labeled: bool = False) -> Optional[str]:
    """
    Extract YYYY-MM-DD from a shard path if it matches the expected naming.
    """
    name = os.path.basename(path)
    regex = LABELED_FILE_REGEX if labeled else RAW_FILE_REGEX
    m = regex.match(name)
    return m.group(1) if m else None

def shard_output_path_for_date(date_str: str, labeled: bool = False) -> str:
    """
    Build a shard path inside logs/ for a given date and type (raw/labeled).
    """
    ensure_logs_dir()
    if labeled:
        fname = f"nexto_clears_labeled_{date_str}.jsonl"
    else:
        fname = f"nexto_clears_{date_str}.jsonl"
    return os.path.join(LOGS_DIR, fname)

def deduped_output_path_for_date(date_str: str) -> str:
    """
    Build a deduplicated shard path inside logs/ for a given date.
    """
    ensure_logs_dir()
    fname = f"nexto_clears_deduped_{date_str}.jsonl"
    return os.path.join(LOGS_DIR, fname)

def iter_json_records(paths: Iterable[str]) -> Generator[dict, None, None]:
    """
    Stream JSONL records from a list of shard paths.
    """
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    # Skip malformed lines
                    continue

def count_lines(path: str) -> int:
    """
    Count JSONL lines in a file.
    """
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n
