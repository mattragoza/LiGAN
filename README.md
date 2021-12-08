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

## Usage

### Generating molecules

To generate molecules, you must first download the pretrained model weights:

```
wget https://bits.csb.pitt.edu/files/train6_CVAE2_0_p0_4.0_4.0_k200_d_1.6_r0_n_4.0_65_iter_1000000.gen_model_state -P weights/
```

Then just run the `generate.py` script with the default configuration file:

```
python3 generate.py config/generate.config
```

### Training a model

To train a model from scratch, you must first download the full Crossdocked2020 data set:

```
wget https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.1.tgz -P data/
wget https://bits.csb.pitt.edu/files/
```

More info about this data set can be found [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020).

Then you can run the `train.py` script with the default configuration file:

```
python3 train.py config/train.config
```

## Citation

Coming soon!
