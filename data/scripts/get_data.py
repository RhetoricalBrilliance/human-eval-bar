import asyncio
import os
import random
import httpx
import sys

from archive_utils import parse_archive_month, filter_archives_in_range
from pgn_utils import convert_pgn_to_fen
from api_utils import get_top_blitz_usernames, get_player_archives, fetch_archive_pgn, bounded_gather
from normalize import normalize_dataset
from config import PGN_PATH, TIMEOUT, MAX_CONCURRENCY, FEN_PATH, TOTAL_USERNAMES, NORMALIZED_PATH
# print('Sys:', sys.executable)
async def main(): 
    normalize_dataset(FEN_PATH, NORMALIZED_PATH)
    return

    os.makedirs(os.path.dirname(PGN_PATH), exist_ok=True)
    
    async with httpx.AsyncClient(timeout=TIMEOUT, http2=True) as client: 
        print("Fetching top 50 blitz players...")
        top_usernames = await get_top_blitz_usernames(client, upto=TOTAL_USERNAMES)
        print("Found top 50 blitz players.")
        
        out = open(PGN_PATH, "w")
        
        try: 
            for idx, username in enumerate(top_usernames, start=1):
                print(f"[{idx:02d}/50] {username}: listing archives…")
                archives = await get_player_archives(client, username)
                if not archives:
                        print(f"No archives for {username}. Skipping.")
                        continue
                archives = filter_archives_in_range(archives)
                
                unique_archives = sorted(set(archives), key=lambda u: parse_archive_month(u))
                tasks = [fetch_archive_pgn(client, a) for a in unique_archives]
                results: list[str] = await bounded_gather((t for t in tasks), MAX_CONCURRENCY)
                
                for pgn_block in results: 
                    if pgn_block.strip() and "Let's Play!" not in pgn_block and "Play vs Coach" not in pgn_block: 
                        out.write(pgn_block.strip() + "\n\n")
                    print(f"Saved {len(results)} PGN archives for {username}")
                await asyncio.sleep(0.2 + random.random() * 0.3)
        finally: 
            out.close()
            
    print("[PGNS COMPLETE] Completed all archives. Converting PGNs to FENs...")
            
    convert_pgn_to_fen(PGN_PATH, FEN_PATH)

    print(f"[FEN COMPLETE]. Normalizing dataset...")
    
    
    print(f"[NORMALIZING COMPLETE], Dataset is complete!")
    
if __name__ == "__main__": 
    asyncio.run(main())    
