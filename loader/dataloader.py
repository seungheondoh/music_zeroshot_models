import pickle
from typing import Callable, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .dataset import MSD_Dataset


class DataPipeline(LightningDataModule):
    def __init__(self, data_dir, msd_dir, task_type, emb_type, supervisions, duration, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = MSD_Dataset
        self.data_dir = data_dir
        self.msd_dir = msd_dir
        self.task_type = task_type
        self.emb_type = emb_type
        self.supervisions = supervisions
        self.duration = duration
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.data_dir,
                self.msd_dir,
                "TRAIN",
                self.task_type,
                self.emb_type,
                self.supervisions,
                self.duration,
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.data_dir,
                self.msd_dir,
                "VALID",
                self.task_type,
                self.emb_type,
                self.supervisions,
                self.duration,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.data_dir,
                self.msd_dir,
                "TEST",
                self.task_type,
                self.emb_type,
                self.supervisions,
                self.duration,
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, data_dir, msd_dir, split, task_type, emb_type, supervisions, duration) -> Dataset:
        dataset = dataset_builder(data_dir, msd_dir, split, task_type, emb_type, supervisions, duration)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size, num_workers, drop_last, shuffle, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last = drop_last, 
            shuffle = shuffle,
            **kwargs
        )