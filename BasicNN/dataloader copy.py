import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

RESULT_MAP = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

FEATURE_COLS = [
    "WhiteElo","BlackElo","Start Time","Increment","move_number",
    "norm_white_clock","norm_black_clock", 
    "sf_cp", "sf_w", "sf_d"
    # add "sf_cp","sf_w","sf_d","sf_l" if present
]

class ParquetChessDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path, columns=FEATURE_COLS + ["Result"])
        df = df[df["Result"].isin(RESULT_MAP)]  # defensive

        X = df[FEATURE_COLS].astype("float32").to_numpy(copy=False)
        y = df["Result"].map(RESULT_MAP).astype("int64").to_numpy(copy=False)

        # Store as torch tensors for zero-copy reads in __getitem__
        self.X = torch.from_numpy(X).contiguous()
        self.y = torch.from_numpy(y)

    def __len__(self): 
        return self.X.shape[0]
    
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

def get_dataloader(parquet_path, batch_size=128, shuffle=True, pin_memory=False):
    ds = ParquetChessDataset(parquet_path)
    # data already in memory: num_workers=0 is typically best
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin_memory)