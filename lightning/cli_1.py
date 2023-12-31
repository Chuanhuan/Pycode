# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import BoringDataModule, DemoModel
import torch

class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()


class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)

class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)


class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)

class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
    def step(self):
        print("⚡", "using LitLRScheduler", "⚡")
        super().step()

cli = LightningCLI(model_class=DemoModel, datamodule_class=BoringDataModule)

# # use Model1
# python main.py fit --model Model1
#
# # use Model2
# python main.py fit --model Model2
