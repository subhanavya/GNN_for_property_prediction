import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Batch


class AttentiveFPModel(pl.LightningModule):
    def __init__(
        self,
        props,
        hidden_dim=128,            # from params.yaml
        out_dim=1,                 # from params.yaml
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs                   # ignore any extra keys in params.yaml
    ):
        super().__init__()

        # ✅ Save props as well so checkpoint reloading works without errors
        self.save_hyperparameters()

        self.props = props
        n_out = out_dim if out_dim > 0 else max(1, len(props))

        # AttentiveFP GNN
        self.gnn = AttentiveFP(
            in_channels=10,
            hidden_channels=hidden_dim,
            out_channels=n_out,
            edge_dim=1,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch: Batch):
        return self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def training_step(self, batch: Batch, batch_idx: int):
        preds = self(batch).view(-1, len(self.props)) if len(self.props) > 1 else self(batch).view(-1, 1)
        target = batch.y.view_as(preds)
        loss = F.mse_loss(preds, target)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        preds = self(batch).view(-1, len(self.props)) if len(self.props) > 1 else self(batch).view(-1, 1)
        target = batch.y.view_as(preds)
        mae = F.l1_loss(preds, target)
        self.log('val/mae', mae, prog_bar=True, on_epoch=True)
        return mae

    def test_step(self, batch: Batch, batch_idx: int):
        preds = self(batch).view(-1, len(self.props)) if len(self.props) > 1 else self(batch).view(-1, 1)
        target = batch.y.view_as(preds)
        mae = F.l1_loss(preds, target)
        self.log('test/mae', mae, prog_bar=True, on_epoch=True)
        return mae

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        preds = self(batch)
        return preds.detach().cpu()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
