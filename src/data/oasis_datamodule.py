import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from monai.transforms import Compose, RandFlip, RandRotate, ToTensor
from src.data.components.oasis_dataset import OASISDataset
import os


class OASISDataModule(LightningDataModule):
    """`LightningDataModule` for the OASIS dataset (2D slices)."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        csv_path: str = "data/processed/metadata.csv",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `OASISDataModule`.

        :param data_dir: Directory where the preprocessed .npy files are stored.
        :param csv_path: Path to the metadata CSV.
        :param train_val_test_split: The train, validation and test split proportions.
        :param batch_size: The batch size.
        :param num_workers: The number of workers.
        :param pin_memory: Whether to pin memory.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # MONAI transforms for training (data augmentation)
        self.train_transforms = Compose(
            [
                RandFlip(spatial_axis=0, prob=0.5),
                RandRotate(range_x=0.2, prob=0.5),
                ToTensor(),
            ]
        )
        
        # MONAI transforms for validation/test (just tensor conversion)
        self.val_transforms = Compose([ToTensor()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if not self.data_train and not self.data_val and not self.data_test:
            # Create a base dataset to get the length
            full_dataset = OASISDataset(
                csv_path=self.hparams.csv_path,
                data_dir=self.hparams.data_dir
            )
            
            # Use patient-level splitting to prevent data leakage
            import numpy as np
            unique_patients = full_dataset.df['patient_id'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_patients)
            
            total_patients = len(unique_patients)
            if isinstance(self.hparams.train_val_test_split[0], float):
                train_p_size = int(self.hparams.train_val_test_split[0] * total_patients)
                val_p_size = int(self.hparams.train_val_test_split[1] * total_patients)
            else:
                # Fallback if split_lengths were passed as absolute integers (unlikely for patient split, but safe)
                train_p_size = int(self.hparams.train_val_test_split[0])
                val_p_size = int(self.hparams.train_val_test_split[1])
                
            train_patients = unique_patients[:train_p_size]
            val_patients = unique_patients[train_p_size:train_p_size + val_p_size]
            test_patients = unique_patients[train_p_size + val_p_size:]
            
            # Map patient lists back to row indices in the dataframe
            train_indices = full_dataset.df.index[full_dataset.df['patient_id'].isin(train_patients)].tolist()
            val_indices = full_dataset.df.index[full_dataset.df['patient_id'].isin(val_patients)].tolist()
            test_indices = full_dataset.df.index[full_dataset.df['patient_id'].isin(test_patients)].tolist()
            
            # Re-instantiate datasets with specific transforms for each split
            self.data_train = OASISDataset(
                csv_path=self.hparams.csv_path,
                data_dir=self.hparams.data_dir,
                transform=self.train_transforms
            )
            # Filter by indices
            self.data_train.df = self.data_train.df.iloc[train_indices].reset_index(drop=True)
            
            self.data_val = OASISDataset(
                csv_path=self.hparams.csv_path,
                data_dir=self.hparams.data_dir,
                transform=self.val_transforms
            )
            self.data_val.df = self.data_val.df.iloc[val_indices].reset_index(drop=True)
            
            self.data_test = OASISDataset(
                csv_path=self.hparams.csv_path,
                data_dir=self.hparams.data_dir,
                transform=self.val_transforms
            )
            self.data_test.df = self.data_test.df.iloc[test_indices].reset_index(drop=True)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    dm = OASISDataModule()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label: {batch['label']}")
    print(f"Clinical sliders: {batch['clinical_sliders']}")
