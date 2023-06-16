# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Tuple, Union

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class ClassImageBase(ImageFolder):
    def __init__(self, root: str, split: str, transform: Callable) -> None:
        root = Path(root) / split
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)

        return {"image": image, "class": torch.tensor([target])}


class ClassImageTrain(ClassImageBase):
    def __init__(
        self, root: str, resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75
    ) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = T.Compose(
            [T.Resize(resolution), T.RandomCrop(resolution), T.RandomHorizontalFlip(), T.ToTensor()]
        )

        super().__init__(root, "train", transform)


class ClassImageValidation(ClassImageBase):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = T.Compose([T.Resize(resolution), T.CenterCrop(resolution), T.ToTensor()])

        super().__init__(root, "val", transform)
