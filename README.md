# liGAN

![liGAN methods diagram](ligan_methods.png)

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

**NOTE**: Please be aware that the current version of molgrid provided through pip/conda is incompatible with conda openbabel, and you will likely get segmentation faults if you install them both through conda.

A molgrid conda build recipe is in the works (see https://github.com/mattragoza/conda-molgrid), but for now you can use [this environment](https://github.com/mattragoza/conda-molgrid/blob/master/environment.yaml) to build libmolgrid from source.

## Usage

### Generating molecules

To generate molecules, you must first download the pretrained model weights:

```
sh download_weights.sh
```

Then just run the `generate.py` script with the default configuration file:

```
python3 generate.py config/generate.config
```

### Training a model

To train a model from scratch, you must first download the full Crossdocked2020 data set:

```
sh download_data.sh
```

More info about this data set can be found [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020).

Then you can run the `train.py` script with the default configuration file:

```
python3 train.py config/train.config
```

## Citation

Coming soon!
