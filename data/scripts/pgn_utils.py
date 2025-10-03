import chess.pgn
import re
import json
import time
import math
from typing import Tuple, Dict, Iterable, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from stockfish_features import _worker_init, _eval_with_engine

STOCKFISH_PATH = "/usr/local/bin/stockfish"
EVAL_DEPTH = 15
BATCH_SIZE = 3000
PROGRESS_EVERY = 200


def _iter_game_records(pgn_path: str) -> Iterable[Dict]:
    seen_ids = set()
    with open(pgn_path, "r", encoding="utf-8") as f_in:
        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break

            link = game.headers.get("Link", "")
            if not link:
                continue
            game_id = link.split("/")[-1]
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            raw_game = str(game)
            clock_matches = re.findall(r"\[%clk\s([0-9:.]+)\]", raw_game)

            board = game.board()
            start_sec_str, start_sec, increment = time_control_and_increment(game.headers.get("TimeControl"))
            if start_sec <= 60 and increment == 0:
                continue

            metadata = {
                "White": game.headers.get("White"),
                "Black": game.headers.get("Black"),
                "WhiteElo": int(game.headers.get("WhiteElo")),
                "BlackElo": int(game.headers.get("BlackElo")),
                "Start Time": int(start_sec),
                "Result": game.headers.get("Result"),
                "Date": game.headers.get("UTCDate"),
                "Increment": increment
            }

            move_number = 1.5
            clock_index = 0
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    continue

                if clock_index == 0:
                    white_clock = clock_to_sec(clock_matches[0])
                    black_clock = clock_to_sec(start_sec_str)
                else:
                    if clock_index % 2 == 0:
                        white_clock = clock_to_sec(clock_matches[clock_index])
                        black_clock = clock_to_sec(clock_matches[clock_index - 1])
                    else:
                        white_clock = clock_to_sec(clock_matches[clock_index - 1])
                        black_clock = clock_to_sec(clock_matches[clock_index])

                yield {
                    **metadata,
                    "fen": board.fen(),
                    "move_number": move_number,
                    "white_clock": white_clock,
                    "black_clock": black_clock,
                    "norm_white_clock": round(white_clock / start_sec, 2),
                    "norm_black_clock": round(black_clock / start_sec, 2),
                }

                clock_index += 1
                move_number += 0.5


def count_positions(pgn_path: str) -> int:
    """
    Fast pre-count pass using the same filters as _iter_game_records,
    but without building dicts or running Stockfish.
    """
    seen_ids = set()
    total = 0
    with open(pgn_path, "r", encoding="utf-8") as f_in:
        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break

            link = game.headers.get("Link", "")
            if not link:
                continue
            game_id = link.split("/")[-1]
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            raw_game = str(game)
            clock_matches = re.findall(r"\[%clk\s([0-9:.]+)\]", raw_game)

            board = game.board()
            start_sec_str, start_sec, increment = time_control_and_increment(game.headers.get("TimeControl"))
            if start_sec <= 60 and increment == 0:
                continue

            clock_index = 0
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    continue

                # We also ensure there are clocks to index into (defensive)
                if clock_index == 0 and not clock_matches:
                    break

                total += 1
                clock_index += 1
    return total


def convert_pgn_to_fen(pgn_path: str, output_file: str):
    t0 = time.time()

    # Pre-count expected positions so we can show total batches & ETA
    total_expected = count_positions(pgn_path)
    batches_expected = math.ceil(total_expected / BATCH_SIZE) if total_expected else 0
    print(f"[INFO] Expecting ~{total_expected} positions across ~{batches_expected} batches "
          f"(batch size {BATCH_SIZE}). Depth={EVAL_DEPTH}")

    cpu_count = multiprocessing.cpu_count()
    print(f"[INFO] Parallel eval on {cpu_count} processes (Threads=1 per engine).")

    def _batches(src_iter: Iterable[Dict], size: int):
        buf: List[Dict] = []
        for rec in src_iter:
            buf.append(rec)
            if len(buf) >= size:
                yield buf
                buf = []
        if buf:
            yield buf

    total_written = 0
    total_submitted = 0

    with open(output_file, "w", encoding="utf-8") as f_out, \
         ProcessPoolExecutor(
             max_workers=cpu_count,
             initializer=_worker_init,
             initargs=(STOCKFISH_PATH, 1, 256, True)
         ) as pool:

        for b_idx, batch in enumerate(_batches(_iter_game_records(pgn_path), BATCH_SIZE), start=1):
            future_to_rec = {}
            for rec in batch:
                fut = pool.submit(_eval_with_engine, rec["fen"], EVAL_DEPTH)
                future_to_rec[fut] = rec
            total_submitted += len(future_to_rec)

            for fut in as_completed(future_to_rec):
                res = fut.result()
                if res is None:
                    continue
                sf_cp, w, d, l = res
                rec = future_to_rec[fut]
                rec["sf_cp"], rec["sf_w"], rec["sf_d"], rec["sf_l"] = sf_cp, w, d, l
                f_out.write(json.dumps(rec) + "\n")
                total_written += 1

                if total_written % PROGRESS_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = total_written / max(elapsed, 1e-9)
                    remaining = max(total_expected - total_written, 0)
                    eta = remaining / max(rate, 1e-9) if total_expected else float("nan")
                    print(f"[PROGRESS] {total_written}/{total_expected} "
                          f"({rate:.1f}/s). Batch {b_idx}/{batches_expected}. "
                          f"Elapsed {elapsed:.1f}s, ETA {eta/60:.1f} min")

    print(f"[DONE] Wrote {total_written} / {total_expected} positions to {output_file} "
          f"in {time.time()-t0:.1f}s")


def time_control_and_increment(tc: str) -> Tuple[str, int, int]:
    s = str(tc).strip()
    m = re.match(r'^\s*(?P<base>\d+(?:\.\d+)?)\s*(?:\+\s*(?P<inc>\d+(?:\.\d+)?))?\s*$', s)
    base_secs = int(round(float(m.group('base')))) if m else 0
    inc_secs = int(round(float(m.group('inc')))) if (m and m.group('inc')) else 0
    h, rem = divmod(base_secs, 3600)
    mm, ss = divmod(rem, 60)
    clock_str = f"{h}:{mm:02d}:{ss:02d}"
    return clock_str, base_secs, inc_secs


def clock_to_sec(clock: str) -> int:
    parts = clock.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected time format: {clock}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds







#### NON PARALLELIZED VERSION ###
# import chess.pgn
# import re
# import json
# from typing import Tuple
# from stockfish_features import add_sf_to_record, make_stockfish_engine, basic_eval
# import time


# STOCKFISH_PATH = "/usr/local/bin/stockfish"
# EVAL_DEPTH = 15
# BATCH_SIZE = 5000

# def convert_pgn_to_fen(pgn_path: str, output_file: str): 
#     """
#     Convert a PGN file of many PGNS into a JSONL dataset of FEN's. 
#     Each data point has: 
#         metadata (white/black elo, time control, result)
#         moves (list of board positions and white/black turns, as well as white/black remaining times)
#     Avoids duplicate moves
#     """
#     t0 = time.time()
#     engine = make_stockfish_engine(STOCKFISH_PATH, threads=4, hash_mb=1024, show_wdl=True)


#     all_positions = []
#     seen_ids = set() # Track which game IDs have been seen, to avoid duplicates
#     positions = 0
#     with open (pgn_path, "r", encoding="utf-8") as f_in, \
#         open(output_file, "w", encoding="utf-8") as f_out: 
        
#         game_idx = 0
#         while True: 
#             game = chess.pgn.read_game(f_in)
#             if game is None: 
#                 break
#             game_idx += 1
            
#             game_id = None
#             link = game.headers["Link"]
#             game_id = link.split("/")[-1] # This contains the game ID

#             if game_id in seen_ids: # Ignore already seen games
#                 continue 
#             seen_ids.add(game_id)
            
#             raw_game = str(game)
#             clock_matches = re.findall(r"\[%clk\s([0-9:.]+)\]", raw_game)
#             # print(f"Clock matches: {clock_matches}")
#             # print("Length of clock matches:", len(clock_matches))
            
#             board = game.board()
#             start_sec_str, start_sec, increment = time_control_and_increment(game.headers.get("TimeControl"))
#             if start_sec <= 60 and increment == 0:
#                 continue 
    
#             metadata = {
#                 "White": game.headers.get("White"),
#                 "Black": game.headers.get("Black"),
#                 "WhiteElo": int(game.headers.get("WhiteElo")),
#                 "BlackElo": int(game.headers.get("BlackElo")),
#                 "Start Time": int(start_sec), 
#                 "Result": game.headers.get("Result"), 
#                 "Date": game.headers.get("UTCDate"), 
#                 "Increment": increment
#             }
            
#             move_number = 1.5
#             clock_index = 0
            
#             # print("Length of Game mainline moves:", game.mainline_moves())
            
#             for move in game.mainline_moves(): 
#                 board.push(move)
#                 if board.is_game_over():
#                     continue
#                 white_clock, black_clock = None, None
#                 if clock_index == 0:  # Different case for beginning of game, mark black as having full time after White's first move
#                         white_clock = clock_to_sec(clock_matches[0])
#                         black_clock = clock_to_sec(start_sec_str)
#                         norm_white_clock = round(white_clock / start_sec, 2)
#                         norm_black_clock = round(black_clock / start_sec, 2)

#                 else:
#                     if clock_index % 2 == 0: 
#                         white_clock = clock_to_sec(clock_matches[clock_index])
#                         black_clock = clock_to_sec(clock_matches[clock_index - 1])
#                         norm_white_clock = round(white_clock / start_sec, 2)
#                         norm_black_clock = round(black_clock / start_sec, 2)
#                     else: 
#                         white_clock = clock_to_sec(clock_matches[clock_index - 1])
#                         black_clock = clock_to_sec(clock_matches[clock_index])
#                         norm_white_clock = round(white_clock / start_sec, 2)
#                         norm_black_clock = round(black_clock / start_sec, 2)
                
#                 clock_index += 1
                    
#                 game_record = {**metadata, "fen": board.fen(), "move_number": move_number, "white_clock": white_clock, "black_clock": black_clock, "norm_white_clock": norm_white_clock, "norm_black_clock": norm_black_clock}
#                 add_sf_to_record(game_record=game_record, engine=engine, depth=EVAL_DEPTH)
#                 if positions % 100 == 0:
#                     print(f"Completed {positions} in time {time.time() - t0}")
#                 f_out.write(json.dumps(game_record) + "\n")
#                 all_positions.append(game_record) 
#                 positions += 1
#                 move_number += 0.5
    
#     print(f"[FENs CREATED] Wrote {len(all_positions)} positions to {output_file} from {len(seen_ids)} unique games.")
    
    
# def time_control_and_increment(tc: str) -> Tuple[str, int, int]: 
#     """Convert a time control into the standard PGN format for time, as to mark Black's starting time. Returns increment if exists."""
#     s = str(tc).strip()
#     # Matches TimeControl+Increment format used in Chesscom PGN, base = TimeControl
#     m = re.match(r'^\s*(?P<base>\d+(?:\.\d+)?)\s*(?:\+\s*(?P<inc>\d+(?:\.\d+)?))?\s*$', s) 
    
    
#     base_secs = int(round(float(m.group('base')))) # Starting time
#     inc_secs  = int(round(float(m.group('inc')))) if m.group('inc') is not None else 0 # Increment
    
#     # Calculate precise time control
#     h, rem = divmod(base_secs, 3600)
#     mm, ss = divmod(rem, 60)
#     clock_str = f"{h}:{mm:02d}:{ss:02d}"
#     return clock_str, base_secs, inc_secs

# def clock_to_sec(clock: str) -> int: 
    """Convert clock string into seconds."""
    parts = clock.split(":")
    if len(parts) != 3: 
        raise ValueError(f"Unexpected time format: {clock}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds 
    return total_seconds