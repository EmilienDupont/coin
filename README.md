# COIN 🌟

This repo contains a Pytorch implementation of [COIN: COmpression with Implicit Neural representations](https://arxiv.org/abs/2103.03123), including code to reproduce all experiments and plots in the paper.

<img src="https://github.com/EmilienDupont/coin/raw/main/imgs/coin_summary.png" width="800">

## Requirements

We ran our experiments with `python 3.8.7` using `torch 1.7.0` and `torchvision 0.8.0` but the code is likely to work with earlier versions too. All requirements can be installed with

```pip install -r requirements.txt```

## Usage

### Compression

To compress the image `kodak-dataset/kodim15.png`, run

```python main.py -ld logs_dir```

This will save the COIN model and the reconstruction of the image (as well as logs of the losses and PSNR) to the `logs_dir` directory. To run on a specific image in the Kodak dataset, add the `-iid` flag. For example, to compress image 3, run

```python main.py -ld logs_dir -iid 3```

To compress the entire Kodak dataset, run

```python main.py -ld logs_dir -fd```

**NOTE**: The half precision version of `torch.sin` is only implemented in CUDA, so the half precision models can only be run on GPU,
you need that to reproduce the results from the paper.

To reproduce the results from the paper, run the architectures listed in Appendix A
```
python main.py -ld logs_dir -fd --num_layers 5 --layer_size 20
python main.py -ld logs_dir -fd --num_layers 5 --layer_size 30
python main.py -ld logs_dir -fd --num_layers 10 --layer_size 28
python main.py -ld logs_dir -fd --num_layers 10 --layer_size 40
python main.py -ld logs_dir -fd --num_layers 13 --layer_size 49
```

### Plots

To recreate plots from the paper, run

```python plots.py```

See the `plots.py` file to customize plots.

## Acknowledgements

Our benchmarks and plots are based on the [CompressAI](https://github.com/InterDigitalInc/CompressAI) library. Our SIREN implementation is based on [lucidrains'](https://github.com/lucidrains/siren-pytorch) implementation.

## License

MIT
