from torch import optim, nn
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.save_hyperparameters()

    def forward(self, x):
        x_hat = self.GRU(x)

        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer