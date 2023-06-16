from typing import Callable, Optional, Tuple, Union

from datasets import load_dataset
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..utils.general import initialize_from_config


class HFDatasetBase(Dataset):
    def __init__(
        self,
        repo_id: str,
        split: str,
        tokenizer: Optional[OmegaConf] = None,
        transform: Optional[Callable] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self._streaming = streaming
        self.dataset: Dataset = load_dataset(repo_id, split=split, streaming=streaming)
        self.tokenizer = initialize_from_config(tokenizer) if tokenizer is not None else None
        self.transform = transform

        if streaming:
            self._length = self.dataset.info.splits[split].num_examples

    def __len__(self) -> int:
        if self._streaming:
            return self._length
        if hasattr(self.dataset, "num_rows"):
            return self.dataset.num_rows
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        image: Image.Image = sample["image"]
        if self.transform is not None:
            image = self.transform(image)

        caption: str = " ".join(sample["caption"])
        if self.tokenizer is not None:
            caption = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True)
            return {"image": image, "caption": caption}
        else:
            return {"image": image}


class HFDatasetTrain(HFDatasetBase):
    def __init__(
        self,
        repo_id: str,
        resolution: Union[Tuple[int, int], int] = 256,
        tokenizer: Optional[OmegaConf] = None,
        streaming: bool = False,
    ) -> None:
        transform = T.Compose(
            [
                T.Resize(resolution),
                T.RandomCrop(resolution),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        super().__init__(repo_id, "train", tokenizer, transform, streaming)


class HFDatasetValidation(HFDatasetBase):
    def __init__(
        self,
        repo_id: str,
        resolution: Union[Tuple[int, int], int] = 256,
        tokenizer: Optional[OmegaConf] = None,
        streaming: bool = False,
    ) -> None:
        transform = T.Compose(
            [
                T.Resize(resolution),
                T.CenterCrop(resolution),
                T.ToTensor(),
            ]
        )
        super().__init__(repo_id, "test", tokenizer, transform, streaming)
