# Investigating the Benefits of Projection Head for Representation Learning

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This is fork of a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). A study and experimental investigation of projection heads in self-supervised learning, reproducing and extending the results of the paper [Investigating the Benefits of Projection Head for Representation Learning](https://arxiv.org/abs/2403.11391v1)

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Examples

To train a ResNet on STL10 dataset, run:

```bash
python3 train.py model=resnet
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
