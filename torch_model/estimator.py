import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as f
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchmetrics.classification import MulticlassF1Score


class NewsClassificator(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.n_classes = num_classes
        self.max_features = 142665
        self.embed_size = 300
        self.hidden_size = 128
        self.drp = 0.1
        self.f1_score = MulticlassF1Score(num_classes=self.n_classes, average='macro')

        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.drp)
        self.dropout1 = nn.Dropout(self.drp)
        self.linear1 = nn.Linear(self.hidden_size * 2, 64)
        self.dropout2 = nn.Dropout(self.drp)
        self.out = nn.Linear(64, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, emb):
        emb = emb.to(torch.float32)
        emb, _ = self.lstm(emb)
        emb = self.dropout1(emb)
        emb = f.relu(self.linear1(emb))
        emb = self.dropout2(emb)
        out = self.out(emb)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        output = self(x)
        loss = self.criterion(output, y)
        f1_train = self.f1_score(output, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "f1": f1_train}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        output = self(x)
        loss = self.criterion(output, y)
        f1_valid = self.f1_score(output, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_f1": f1_valid}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["f1"] for x in outputs]).mean()

        print(f"| Train_f1: {avg_f1:.2f}, Train_loss: {avg_loss:.2f}")

        self.log("train_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_f1", avg_f1, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

        print(f"[Epoch {self.trainer.current_epoch:3}] Val_f1: {avg_f1:.2f}, Val_loss: {avg_loss:.2f}", end=" ")

        self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", avg_f1, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)

        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                     first_cycle_steps=1800,
                                                     cycle_mult=1.0,
                                                     max_lr=0.1,
                                                     min_lr=0.00000005,
                                                     warmup_steps=400,
                                                     gamma=0.7)

        return [optimizer], [lr_scheduler]
