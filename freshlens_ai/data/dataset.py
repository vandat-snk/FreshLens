"""PyTorch Dataset for locked FreshLens image records."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from freshlens_ai.data.image_io import read_record_image
from freshlens_ai.data.transforms import image_transform


class FruitDataset(Dataset):
    def __init__(
        self,
        root,
        rows,
        training=False,
    ):
        self.root = Path(root)
        self.rows = list(rows)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        # Hash and decode the same bytes on each read,
        # including during resume.
        image = read_record_image(
            self.root,
            row,
        )

        return (
            self.transform(image),
            row["target"],
            index,
        )
