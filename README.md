# liGAN

liGAN is a PyTorch project for structure-based drug discovery with deep generative models of atomic density grids.

- [CVAE journal paper](https://arxiv.org/abs/2110.15200) (under review)

- [CVAE NeurIPS 2020 workshop paper](https://arxiv.org/abs/2010.14442) ([15 minute talk](https://youtu.be/zru1FqCd8Ks))

- [VAE NeurIPS 2020 workshop paper](https://arxiv.org/abs/2010.08687) ([2 minute talk](https://youtu.be/Pyc6xwtGaUM))

## Dependencies

- numpy
- pandas
- scikit-image
- torch
- openbabel
- rdkit
- [molgrid](https://github.com/gnina/libmolgrid)
- [gnina](https://github.com/gnina/gnina)

## Usage

To start generating molecules, just run the `generate.py` scripts with a configuration file:

```
python3 generate.py config/generate.config
```

## Citation

Coming soon!
