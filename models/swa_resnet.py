import torch
import torch.nn.functional as F

from models.lit_resnet import LitResnet
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

class SWAResnet(LitResnet):
    def __init__(self, trained_model, lr=0.01):
        super().__init__()

        self.save_hyperparameters("lr")
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)


    def forward(self, x):
        out = self.swa_model(x)
        return F.log_softmax(out, dim=1)


    def training_epoch_end(self, training_step_outputs):
        self.swa_model.update_parameters(self.model)


    def validation_step(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer


    def on_train_end(self):
        update_bn(self.trainer.datamodule.train_dataloader(), self.swa_model, device=self.device)