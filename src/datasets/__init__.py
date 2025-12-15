from src.datasets.cifar import CIFARDataset
from src.datasets.stl import STLDataset
from src.datasets.rotated_stl import RotatedSTLDataset
from src.datasets.mnist import MNISTDataset
from src.datasets.mnist_on_cifar import MNISTonCIFARDataset

__all__ = ["CIFARDataset", "STLDataset", "RotatedSTLDataset", "MNISTDataset", "MNISTonCIFARDataset"]