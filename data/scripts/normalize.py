import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import json
from pathlib import Path

# Columns we never include in modeling
EXCLUDE = {"White", "Black", "Date", "fen", "Result"}

# Columns we keep numeric but do NOT scale
DONT_SCALE = {"norm_white_clock", "norm_black_clock", "sf_w", "sf_d", "sf_l"}

def load_jsonl(path: str):
    """Load JSONL into a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def normalize_dataset(raw_path: str, normalized_path: str) -> Tuple[pd.DataFrame, StandardScaler, list]:
    """
    Normalize numeric features for training purposes.
    Excludes categorical/target columns and skips already-normalized/probability columns.

    Returns
    -------
    df_norm : pd.DataFrame
        DataFrame with scaled numeric features (except excluded ones).
    scaler : StandardScaler
        Fitted scaler you can reuse on val/test.
    scaled_cols : list[str]
        Columns that were scaled (so you can apply the same transform later).
    """
    df = pd.DataFrame(load_jsonl(raw_path))

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Which to scale (not in EXCLUDE or DONT_SCALE)
    scaled_cols = [c for c in numeric_cols if c not in EXCLUDE and c not in DONT_SCALE]

    # Check for missing values
    if df[scaled_cols].isna().any().any():
        missing = df[scaled_cols].isna().sum()
        raise ValueError(f"Found missing values in: {missing[missing>0].to_dict()}")

    # Ensure float type
    df[scaled_cols] = df[scaled_cols].astype("float64")

    # Scale only selected columns
    scaler = StandardScaler()
    df.loc[:, scaled_cols] = scaler.fit_transform(df[scaled_cols].values)

    # Save to parquet
    p = Path(normalized_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

    # Optional: visualization
    visualize_dataset(normalized_path, scaled_cols)

    return df, scaler, scaled_cols

def visualize_dataset(normalized_path: str, scaled_cols: list):
    """Quick check that scaled columns look standardized."""
    df = pd.read_parquet(normalized_path)  # needs: pip install pyarrow

    print(df.shape)
    print(df.dtypes.head(20))
    print(df.head(5))

    # Sanity: no NaNs in scaled cols
    print(df[scaled_cols].isna().sum())

    # Mean ~0, std ~1
    print("means:\n", df[scaled_cols].mean().round(3))
    print("stds:\n",  df[scaled_cols].std(ddof=0).round(3))  # population std
