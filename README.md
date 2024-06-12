# EM-mixtures

WIP

## Requirements

- Python 3.8 or greater with Numpy
- [PyTorch](https://pytorch.org/get-started/locally/)
- (optional) Nvidia GPU to make use of PyTorch CUDA (not supported on Mac)

## Usage

`python mixtures.py [some-logliks].tsv`

or

`python mixtures2.py [some-logliks].tsv [another-logliks].tsv` to see the true speed of GPU implementation on the second file and run since the first GPU iteration is slow due to some overhead/warmup.
