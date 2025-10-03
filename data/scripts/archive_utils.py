from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import urlparse
from typing import Iterable
from config import START_MONTH, END_MONTH

def month_iter(start: datetime, end_inclusive: datetime) -> Iterable[datetime]:
    """Yield first-of-month datetime objects from start to end inclusive."""
    d = start.replace(day=1)
    last = end_inclusive.replace(day=1)
    while d <= last:
        yield d
        d += relativedelta(months=1)

def parse_archive_month(archive_url: str) -> tuple[int, int]: 
    """Extract year, month from an archive URL. Only used for validating the given URL to prevent later crashes."""
    parts = urlparse(archive_url).path.strip("/").split("/")
    year = int(parts[-2])
    month = int(parts[-1])
    return year, month

def filter_archives_in_range(archives: list[str]) -> list[str]:
    """Keep only archive URLs that fall in the given time period."""
    months_set = {(m.year, m.month) for m in month_iter(START_MONTH, END_MONTH)}
    kept: list[str] = []
    for a in archives: 
        try:
            y, m = parse_archive_month(a)
        except Exception: 
            continue # skip malformed URLs
        if (y, m) in months_set: # keep only if in desired range
            kept.append(a)
    return kept