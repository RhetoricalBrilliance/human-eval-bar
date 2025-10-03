import asyncio
import httpx
import random
from typing import Iterable
from config import API_BASE, HEADERS, MAX_RETRIES


async def fetch_text(client: httpx.AsyncClient, url: str) -> str | None: 
    """Fetch raw PGN with retry and backoff handling included."""
    backoff = 1.0
    for _ in range(MAX_RETRIES): 
        try: 
            r = await client.get(url, headers=HEADERS)
            if r.status_code == 200: 
                return r.text
            if r.status_code == 301 and r.headers.get("Location"): 
                url = r.headers["Location"]
                continue
            if r.status_code in (429, 503):  # Too many requests/service unavailable 
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 16)
                continue
            if r.status_code == 404: 
                return None
            await asyncio.sleep(backoff) # Generic retry for other errors
            backoff = min(backoff * 2, 16)
        except httpx.RequestError:       # Network error
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 16)
            return None   # Return None if all retries failed
        
async def get_top_blitz_usernames(client: httpx.AsyncClient, upto=50) -> list[str]:
    """Fetch current Top-50 blitz player usernames from leaderboard API"""
    url = f"{API_BASE}/leaderboards"
    data = await client.get(url, headers=HEADERS)
    data = data.json()
    players = sorted(data['live_blitz'], key=lambda p: p.get('rank', 999))  # Sort data by rank in blitz chess
    print('Player 0:', players[0]) 
    return [p['username'] for p in players[:upto]] # Extract top specified usernames (50 max)

async def get_player_archives(client: httpx.AsyncClient, username: str) -> list[str]:
    """Get all archive URLs (each month) for a given players"""
    url = f"{API_BASE}/player/{username.lower()}/games/archives"
    r = await client.get(url, headers=HEADERS)
    if r.status_code != 200: 
        return []
    data = r.json()
    return data.get("archives", [])

async def fetch_archive_pgn(client: httpx.AsyncClient, archive_url: str) -> str: 
    """Fetch PGN for given monthly archive (raw PGN string)."""
    url = archive_url + "/pgn"
    text = await fetch_text(client, url)
    return text or ""

async def bounded_gather(tasks: Iterable[asyncio.Task], limit: int) -> list[str]:
    """Run tasks concurrently bounded by a semaphore."""
    sem = asyncio.Semaphore(limit)
    results: list[str] = []
    
    async def runner(coro): 
        async with sem: 
            return await coro
    
    wrapped = [asyncio.create_task(runner(t)) for t in tasks] # Wrap all tasks
    for w in asyncio.as_completed(wrapped): # Process tasks as they finish
        results.append(await w) # Collect results
        
    return results