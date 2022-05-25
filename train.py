from torch.cuda import is_available
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from models.lit_resnet import LitResnet
from models.swa_resnet import SWAResnet
from data.data import cifar10_dm
from configs.setup_configs import load_config
from utils.metrics import plot_perf

def main(cfg):
    
    model = LitResnet(lr=0.05)
    epochs = cfg[0]['model_params']['lit_resnet']['max_epochs']
    accelerator = cfg[0]['model_params']['lit_resnet']['accelerator']
    log_dir = cfg[2]['log_params']['save_dir']

    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1 if is_available() else None,
        logger=CSVLogger(save_dir=log_dir),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    plot_perf(trainer)

    # swa_model = SWAResnet(model.model, lr=0.01)
    # swa_model.datamodule = cifar10_dm

    # swa_trainer = Trainer(
    #     max_epochs=20,
    #     accelerator="auto",
    #     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    #     callbacks=[TQDMProgressBar(refresh_rate=20)],
    #     logger=CSVLogger(save_dir="logs/"),
    # )

    # swa_trainer.fit(swa_model, cifar10_dm)
    # swa_trainer.test(swa_model, datamodule=cifar10_dm)


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)