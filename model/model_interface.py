



import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # isintance(model, pl.LightningModule) to check the model is from which line
        if isinstance(self.model, pl.LightningModule):
            # dispatch the input batch (audio_version)
            audio_features, targets = batch
            # obtaining output from model
            output = self.model(audio_features)
            # calculate loss
            loss = self.loss_function(output, targets)
            # record into logs
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            return self.model.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # isintance(model, pl.LightningModule) to check the model is from which line
        if isinstance(self.model, pl.LightningModule):
            # dispatch the input batch (audio_version)
            audio_features, targets = batch
            # obtaining output from model
            output = self.model(audio_features)
            # calculate loss
            loss = self.loss_function(output, targets)
            # record into logs
            # we can add more observations on this part (also considering validation_step_end())
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            return self.model.validation_step(self, batch, batch_idx)

    def on_validation_batch_end(self):
        # print the progress bar or print out other information
        self.print('')

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def load_model(self):
        name = self.hp
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package = __package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Model Name {name}.{camel_name}'
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args_new = {}
        for arg in class_args:
            if arg in inkeys:
                args_new[arg] = getattr(self.hparams, arg)
        args_new.update(other_args)
        return Model(**args_new)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

# class TorchNet(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

# class PLNet(pl.LightningModule):
#     def __init__(self) -> None:
#         super().__init__()

# print(1)

