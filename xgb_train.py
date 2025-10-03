"""
xgb_train.py
Train a basic XGBoost (scikit-learn API) classifier on a normalized .parquet file.
- Reads features and 'Result' label
- Train/val/test split with stratification
- Trains XGBClassifier (multiclass)
- Outputs predictions + built-in feature importances (multiple types)
- Saves artifacts: model.json, feature_importance.csv

Usage:
  python xgb_train.py --data data/normalized/normalized.parquet --outdir xgb_out
Optional:
  --features WhiteElo BlackElo ...   # specify feature list explicitly; otherwise uses all columns except 'Result'
  --test_size 0.1 --val_size 0.1 --seed 42
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from xgboost import XGBClassifier

LABEL_MAP = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

def load_data(parquet_path, feature_list=None):
    df = pd.read_parquet(parquet_path)
    if "Result" not in df.columns:
        raise SystemExit("Expected a 'Result' column in the parquet.")
    y = df["Result"].map(LABEL_MAP)
    if y.isna().any():
        bad = df.loc[y.isna(), "Result"].unique()
        raise SystemExit(f"Unmapped labels found in 'Result': {bad}")

    if feature_list is None:
        feature_list = [c for c in df.columns if c != "Result"]
    else:
        missing = [c for c in feature_list if c not in df.columns]
        if missing:
            raise SystemExit(f"Requested features not in data: {missing}")

    X = df[feature_list].astype("float32")
    return X, y.astype("int64"), feature_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to normalized .parquet file with features + 'Result'")
    ap.add_argument("--outdir", default="xgb_out", help="Directory to write outputs")
    ap.add_argument("--features", nargs="*", default=None, help="Optional explicit feature list")
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load
    X, y, feature_list = load_data(args.data, args.features)

    # Train/val/test split (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    val_frac = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=args.seed
    )

    # Model (basic but solid defaults for tabular)
    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",   # fast CPU histogram
        random_state=args.seed,
        n_jobs=-1,
        eval_metric=["mlogloss", "merror"]
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Eval
    def evaluate(split_name, Xs, ys):
        prob = model.predict_proba(Xs)
        pred = prob.argmax(axis=1)
        acc = accuracy_score(ys, pred)
        ll  = log_loss(ys, prob, labels=[0,1,2])
        print(f"{split_name:>5} | acc={acc:.4f} | logloss={ll:.4f}")
        return acc, ll, pred, prob

    print("Evaluation:")
    tr_acc, tr_ll, tr_pred, tr_prob = evaluate("train", X_train, y_train)
    va_acc, va_ll, va_pred, va_prob = evaluate("  val", X_val, y_val)
    te_acc, te_ll, te_pred, te_prob = evaluate(" test", X_test, y_test)

    # Reports
    print("\nTest classification report:")
    print(classification_report(y_test, te_pred, digits=4))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, te_pred))

    # --- Built-in feature importance ---
    # 1) Sklearn-style importance (uses gain by default inside xgboost)
    sk_importance = model.feature_importances_  # shape [n_features]

    # 2) Booster-level importance with different definitions
    booster = model.get_booster()
    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    imp_frames = []
    for itype in importance_types:
        score_dict = booster.get_score(importance_type=itype)  # keys in training order
        # Map back to column names
        # XGBoost uses f{index} in the order it received columns
        scores = np.zeros(len(feature_list), dtype=float)
        for i, col in enumerate(feature_list):
            scores[i] = score_dict.get(f"f{i}", 0.0)
        imp_frames.append(pd.DataFrame({"feature": feature_list, itype: scores}))
    # Merge all types
    imp_df = imp_frames[0]
    for k in range(1, len(imp_frames)):
        imp_df = imp_df.merge(imp_frames[k], on="feature", how="outer")
    # Add sklearn-style
    imp_df["sklearn_importance"] = sk_importance
    imp_df = imp_df.sort_values("gain", ascending=False)

    # Save outputs
    imp_path = Path(args.outdir) / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    model_path = Path(args.outdir) / "xgb_model.json"
    model.save_model(model_path)

    pred_path = Path(args.outdir) / "test_predictions.csv"
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": te_pred,
        "p_white": te_prob[:,0],
        "p_black": te_prob[:,1],
        "p_draw":  te_prob[:,2],
    }).to_csv(pred_path, index=False)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved feature importance to: {imp_path}")
    print(f"Saved test predictions to: {pred_path}")
    

    # ---- SINGLE SOURCE OF TRUTH FOR IMPORTANCE ----
    sk_importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_list,
        "sklearn_importance": sk_importance
    })

    imp_df = imp_df.sort_values("sklearn_importance", ascending=False).reset_index(drop=True)

    # save what is plotted
    imp_path = Path(args.outdir) / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][::-1], imp_df["sklearn_importance"][::-1])  # reverse so largest ends up at the top
    plt.xlabel("Feature Importance (gain-based, normalized)")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(Path(args.outdir) / "feature_importance_plot.png", dpi=150)
    plt.show()

    # sanity print to confirm alignment
    print("\nTop importances:")
    print(imp_df.head(10).to_string(index=False))



if __name__ == "__main__":
    main()
