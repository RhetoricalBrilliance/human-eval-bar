# stockfish_features.py
import atexit
import chess
import chess.engine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Iterable, Optional, Tuple, Union

# -------------------------------
# Per-process Stockfish singleton
# -------------------------------
_ENGINE: Optional[chess.engine.SimpleEngine] = None

def _worker_init(exe_path: str, threads: int = 1, hash_mb: int = 256, show_wdl: bool = True):
    """
    Initializer for each worker process:
    - opens a single Stockfish engine (threads=1 recommended for total throughput),
    - configures it,
    - registers a clean shutdown hook.
    """
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(exe_path)
    cfg = {"Threads": threads, "Hash": hash_mb}
    if show_wdl:
        cfg["UCI_ShowWDL"] = True
    _ENGINE.configure(cfg)

    def _cleanup():
        try:
            _ENGINE.quit()
        except Exception:
            pass

    atexit.register(_cleanup)

def _eval_with_engine(fen: str, depth: int) -> Optional[Tuple[int, float, float, float]]:
    """
    Called inside worker processes. Uses the process-local _ENGINE to evaluate a single FEN.
    Returns (sf_cp, w, d, l) from **White's POV**.
    If the score is a mate in N (either side), returns None (caller can skip).
    """
    assert _ENGINE is not None, "Worker engine not initialized. Did you pass initializer=_worker_init?"
    board = chess.Board(fen)
    info = _ENGINE.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)

    # White-centric eval: + = White better, - = Black better
    p = info["score"].pov(chess.WHITE)
    if p.is_mate():
        return None
    sf_cp = p.score()

    # WDL also White-centric
    wdl = info.get("wdl")
    if wdl:
        # wdl is tuple-like; rotate to White POV then unpack
        wins, draws, losses = wdl.pov(chess.WHITE)
        total = wins + draws + losses
        w = wins / total
        d = draws / total
        l = losses / total
    else:
        w = d = l = None

    return sf_cp, w, d, l


# -------------------------------
# (Optional) single-process helpers
# -------------------------------
def make_stockfish_engine(exe_path: str, threads: int = 1, hash_mb: int = 512, show_wdl: bool = True) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(exe_path)
    config = {"Threads": threads, "Hash": hash_mb}
    if show_wdl:
        config["UCI_ShowWDL"] = True
    eng.configure(config)
    return eng

def basic_eval(fen: str, engine: chess.engine.SimpleEngine, depth: int):
    """Single-process: returns (sf_cp, w, d, l) from White POV."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)
    p = info["score"].pov(chess.WHITE)
    sf_cp = p.score()
    wdl = info.get("wdl")
    if wdl:
        wins, draws, losses = wdl.pov(chess.WHITE)
        total = wins + draws + losses
        w = wins / total
        d = draws / total
        l = losses / total
    else:
        w = d = l = None
    return sf_cp, w, d, l

def add_sf_to_record(game_record: dict, engine: chess.engine.SimpleEngine, depth: int = 15):
    sf_cp, w, d, l = basic_eval(game_record["fen"], engine, depth=depth)
    game_record["sf_cp"] = sf_cp
    game_record["sf_w"] = w
    game_record["sf_d"] = d
    game_record["sf_l"] = l
    return




### OLD STOCKFISH FEATURES.PY ###
# import chess
# import chess.engine
# from typing import Optional, Tuple


# def make_stockfish_engine(exe_path: str, threads=4, hash_mb=512, show_wdl=True) -> chess.engine.SimpleEngine:
#     eng = chess.engine.SimpleEngine.popen_uci(exe_path)
#     config = {"Threads": threads, "Hash": hash_mb}
#     if show_wdl: 
#         config["UCI_ShowWDL"] = True
        
#     eng.configure(config)
#     return eng

# def basic_eval(fen: str, engine: chess.engine.SimpleEngine, depth: int):
#     """Returns stockfish evaluation, w/d/l probabilities for given FEN.""" 
#     board = chess.Board(fen)
#     info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_ALL)
    
#     pov = info["score"].pov(chess.WHITE)
    
#     sf_cp = pov.score()
    
#     wdl = info.get("wdl")
#     if wdl: 
#         wins, draws, losses = wdl.pov(chess.WHITE)
#         total = wins + draws + losses
#         w = wins / total
#         d = draws / total
#         l = losses / total
#     else: 
#         w = d = l = None
        
#     return sf_cp, w, d, l 

# def add_sf_to_record(game_record: dict, engine: chess.engine.SimpleEngine, depth=15): 
#     sf_cp, w, d, l = basic_eval(game_record["fen"], engine, depth=depth)
#     game_record["sf_cp"] = sf_cp
#     game_record["sf_w"] = w
#     game_record["sf_d"] = d
#     game_record["sf_l"] = l
    
#     return
