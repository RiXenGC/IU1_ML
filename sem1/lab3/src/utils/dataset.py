from typing import Any, TypeVar, override
from torch.utils.data import Dataset, Subset
import torch
from torch import Tensor
import numpy as np
from PIL import Image
from pathlib import Path

X = TypeVar("X", covariant=True)
Y = TypeVar("Y", covariant=True)


class RawCovidDataset(Dataset[tuple[Tensor, int]]):
    root: Path
    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[Path, int]]

    def __init__(
        self,
        root_dir: str | Path = "lab3/data/raw",
    ) -> None:
        super().__init__()

        self.root = Path(root_dir)

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        self.class_to_idx = {
            "non-COVID": 0,
            "COVID": 1,
        }

        self.samples = []

        valid_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]

            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix in valid_extensions:
                    self.samples.append((file_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        return image, target


class TransformedDataset(Dataset):
    def __init__(
        self,
        subset: Subset[tuple[Image.Image, int]],
        transform: Any = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, y = self.subset[index]

        if self.transform:
            image = np.array(image)
            x = self.transform(image=image)["image"]
        else:
            x = torch.from_numpy(np.array(image)).float() / 255.0

        return x, y

    def __len__(self) -> int:
        return len(self.subset)
