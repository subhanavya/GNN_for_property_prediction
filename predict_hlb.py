import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.dataloader import SurfProDB
from src.model import AttentiveFPModel
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import json

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with molecules")
    parser.add_argument("--params_yaml", type=str, required=True, help="params.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, default=None, help="If provided, compute metrics")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.params_yaml) as f:
        params = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.input_csv)
    dataset = SurfProDB(df, propnames=[args.target_col] if args.target_col else None)

    # Convert SurfProDB to list of PyG Data objects
    dataset_list = dataset._to_graph_dataset(df)

    # Wrap in DataLoader
    loader = DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False)

    # Load model checkpoint
    ckpt = args.ckpt_path
    if ckpt is None:
        ckpt = os.path.join(args.out_dir, "models", "best.ckpt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError("Best checkpoint not found. Provide --ckpt_path manually.")

    model = AttentiveFPModel.load_from_checkpoint(ckpt, out_dim=1, **params.get("model", {}))

    # Predict
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    preds_list = trainer.predict(model, dataloaders=loader)

    # Concatenate predictions
    preds = np.concatenate([p.numpy() for p in preds_list])

    # Save predictions
    df2 = pd.DataFrame(preds, columns=["preds"])
    pd.concat([df,df2],axis=1).to_csv(
        os.path.join(args.out_dir, "hlb_predictions.csv"), index=False
    )

    # Compute metrics if target column is provided
    if args.target_col:
        y_true = df[args.target_col].values
        metrics = compute_metrics(y_true, preds)
        with open(os.path.join(args.out_dir, "prediction_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
