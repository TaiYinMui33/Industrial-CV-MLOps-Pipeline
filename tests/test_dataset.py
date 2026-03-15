import pytest
from src.data.dataset import DefectDataset

def test_dataset_initialization():
    # Since we don't have real data in CI, we check if it handles empty cases
    dataset = DefectDataset(root_dir=".", split="train")
    assert dataset.split == "train"
    assert len(dataset) == 0 # Mock dataset is empty
