import os
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader
from src.dataloader import SurfProDB
from src.model import AttentiveFPModel
import torch
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def compute_metrics(y_true, y_pred):
    # Flatten both arrays to 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {"mae": mae, "rmse": rmse, "r2": r2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--params_yaml", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_on_full", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    with open(args.params_yaml) as f:
        params = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    db = SurfProDB(df, propnames=["hlb"])
    db.split("KFOLD", n_splits=args.n_splits, val_size=args.val_size)

    all_metrics = []
    best_fold_ckpt = None
    best_fold_mae = float("inf")

    for fold, (train_set, val_set, test_set) in enumerate(zip(db.train, db.valid, db.test)):
        print(f"\n===== Fold {fold+1}/{args.n_splits} =====")
        train_loader = DataLoader(train_set, batch_size=params["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(test_set, batch_size=params["training"]["batch_size"], shuffle=False)

        model = AttentiveFPModel(props=["hlb"], **params.get("model", {}))
        fold_dir = os.path.join(args.out_dir, "models", f"fold{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=fold_dir,
            filename="best",
            save_top_k=1,
            monitor="val/mae",
            mode="min"
        )
        earlystop_cb = EarlyStopping(monitor="val/mae", patience=20, mode="min")

        trainer = pl.Trainer(
            max_epochs=params.get("epochs", 300),
            log_every_n_steps=10,
            callbacks=[checkpoint_cb, earlystop_cb],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1
        )

        trainer.fit(model, train_loader, val_loader)

        # Load best checkpoint for this fold
        best_model = AttentiveFPModel.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            props=["hlb"]
        )

        # Predict on test set
        preds = trainer.predict(best_model, test_loader)
        preds = torch.cat(preds).cpu().numpy()

        y_true = np.concatenate([batch.y.cpu().numpy() for batch in test_loader])
        metrics = compute_metrics(y_true, preds)
        print(f"Fold {fold+1} metrics: {metrics}")

        # Save fold metrics and predictions
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame({"y_true": y_true.flatten(), "y_pred": preds.flatten()}).to_csv(
            os.path.join(fold_dir, "predictions.csv"), index=False
        )

        all_metrics.append(metrics)

        # Track best fold model
        if metrics["mae"] < best_fold_mae:
            best_fold_mae = metrics["mae"]
            best_fold_ckpt = checkpoint_cb.best_model_path

    # Save the best fold checkpoint
    if best_fold_ckpt:
        best_model_out = os.path.join(args.out_dir, "models", "best_fold.ckpt")
        os.makedirs(os.path.dirname(best_model_out), exist_ok=True)
        torch.save(torch.load(best_fold_ckpt), best_model_out)
        print(f"Best fold model saved at: {best_model_out}")

    # CV summary
    summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics])),
                   "std": float(np.std([m[k] for m in all_metrics]))} for k in all_metrics[0].keys()}
    with open(os.path.join(args.out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Retrain on full dataset if requested
    if args.train_on_full:
        print("\n===== Training on full dataset =====")
        full_model = AttentiveFPModel(props=["hlb"], out_dim=1, **params.get("model", {}))
        full_dir = os.path.join(args.out_dir, "models", "full")
        os.makedirs(full_dir, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=full_dir,
            filename="best_full",
            save_top_k=1,
            monitor="val/mae",
            mode="min"
        )
        earlystop_cb = EarlyStopping(monitor="val/mae", patience=20, mode="min")

        full_loader = DataLoader(db._train_full, batch_size=params["training"]["batch_size"], shuffle=True)

        trainer = pl.Trainer(
            max_epochs=params.get("epochs", 300),
            log_every_n_steps=10,
            callbacks=[checkpoint_cb, earlystop_cb],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1
        )

        trainer.fit(full_model, full_loader, full_loader)

        best_full_out = os.path.join(args.out_dir, "models", "best_full.ckpt")
        torch.save(torch.load(checkpoint_cb.best_model_path), best_full_out)
        print(f"Full dataset model saved at: {best_full_out}")


if __name__ == "__main__":
    main()
