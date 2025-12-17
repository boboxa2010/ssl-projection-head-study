import shutil

import numpy as np
import safetensors
import torch
import torchvision
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.datasets.cifar import CIFARDataset
from src.datasets.mnist import MNISTDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MNISTonCIFARDataset(BaseDataset):
    """
    MNIST dataset

    https://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, name="train", s=0.7, *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "mnist_on_cifar" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.

        self.s = s
        self.cifar = CIFARDataset("cifar10")
        self.mnist = MNISTDataset()

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "mnist_on_cifar" / name
        data_path.mkdir(exist_ok=True, parents=True)

        print(f"Parsing MNIST-On-CIFAR Dataset metadata for part {name}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        for i in tqdm(range(min(len(self.cifar), len(self.mnist)))):
            # create dataset
            mnist_item = self.mnist[i]
            cifar_item = self.cifar[i]

            mnist_img, label, _, _ = mnist_item.values()
            cifar_img, _, _, _ = cifar_item.values()
            tr = torchvision.transforms.Resize(16)
            digit = tr(mnist_img)
            mnist_padded = torch.nn.functional.pad(digit, pad=(8, 8, 8, 8, 0, 0))
            mnist_padded = mnist_padded.repeat(3, 1, 1)
            digit_area = (mnist_padded > 0).float()
            cifar_background = cifar_img * (1 - digit_area)
            cifar_foreground = cifar_img * digit_area
            foreground = (1 - self.s) * cifar_foreground + self.s * mnist_padded
            image = cifar_background + foreground

            # save digit also (before was just tensor)
            save_dict = {"tensor": image, "digit": mnist_img}
            save_path = data_path / f"{i:06}.safetensors"
            safetensors.torch.save_file(save_dict, save_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path), "label": label})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
