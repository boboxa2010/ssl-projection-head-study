import shutil

import numpy as np
import safetensors
import safetensors.torch
import torch
import torchvision
from tqdm.auto import tqdm

from torchvision.transforms.functional import rotate

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class RotatedSTLDataset(BaseDataset):
    """
    Rotated STL10 dataset
    """

    def __init__(self, angles: list[int], name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "rotated-stl" / name / "index.json"

        self.angles = angles

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    
    def _generate_rotation(self, img):
        angle = int(np.random.choice(self.angles))
        return rotate(img, angle), self.angles.index(angle)

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
        data_path = ROOT_PATH / "data" / "rotated-stl" / name
        data_path.mkdir(exist_ok=True, parents=True)

        transform = torchvision.transforms.ToTensor()
        stl_data = torchvision.datasets.STL10(
            str(data_path), split=name, download=True, transform=transform
        )

        print(f"Parsing STL Dataset metadata for part {name}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        for i in tqdm(range(len(stl_data))):
            # create dataset
            img, _ = stl_data[i]

            rotated_img, rotated_label = self._generate_rotation(img)

            save_dict = {"tensor": rotated_img}
            save_path = data_path / f"{i:06}.safetensors"
            safetensors.torch.save_file(save_dict, save_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path), "label": rotated_label})

        shutil.rmtree(data_path / "stl10_binary")  # remove

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
