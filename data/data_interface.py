# Copyright 2023 Peipei Wu
#
# CVSSP at University of Surrey, UK
# Supervised by Prof. Wenwu Wang


"""
    Script of interface for different datamodule.
    Requirement:
        Put the instance class of custom dataset under the path of './data'
        The instance class name and file name should follow the rules of camel_name. 
"""


import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import importlib
import inspect

class DInterface(pl.LightningDataModule):
    def __init__(self,
                num_workers = 12,
                dataset = '',
                **kwargs) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = self.instancialize(train=True)
            train_size = int(len(full_dataset) * (self.kwargs[train_size] if 'train_size' in self.kwargs else 0.9))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset=full_dataset, lengths=[train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = self.instancialize(train=False)

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(name),camel_name)
            # getattr(importlib.import_module(name='.'+name,package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}'
            )

    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args_new = {}
        for arg in class_args:
            if arg in inkeys:
                args_new[arg] = self.kwargs[arg]
        args_new.update(other_args)
        return self.data_module(**args_new)
            
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
