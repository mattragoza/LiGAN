# liGAN

![liGAN methods diagram](ligan_methods.png)

liGAN is a PyTorch project for structure-based drug discovery with deep generative models of atomic density grids.

- [Chemical Science article](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a)

- [CVAE NeurIPS 2020 workshop paper](https://arxiv.org/abs/2010.14442) ([15 minute talk](https://youtu.be/zru1FqCd8Ks))

- [VAE NeurIPS 2020 workshop paper](https://arxiv.org/abs/2010.08687) ([2 minute talk](https://youtu.be/Pyc6xwtGaUM))

**NOTE**: You can access the Caffe code that was used for the NeurIPS 2020 workshop by visiting the `caffe` branch of this repo.

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

### Configuration

The two main scripts for using this package are `train.py` and `generate.py`, which both take a config file as the only command line argument. The config files for each script use YAML format to specify settings like what data and model architecture to use.

You can change the input data used by `generate.py` by editting the following config fields in `config/generate.config`:

```yaml
data:
  data_root: <path to directory containing structure files>
  data_file: <path to file that references structure files>
```

The data file should be a CSV-like file where each row is a receptor-ligand pair of structure files formatted like so:

```python
{class_label:d} {affinity:f} {RMSD:f} {receptor_path} {ligand_path}
```

The full structure paths that will be loaded are `data_root/receptor_path` and `data_root/ligand_path`. You can leave the other fields (`class_label`, `affinity`, and `RMSD`) as zero if they are not known, since they are not currently used by the default model.

### Generating molecules

To generate molecules, you must first download the pretrained model weights:

```bash
sh download_weights.sh
```

Then just run the `generate.py` script with the default configuration file:

```bash
python3 generate.py config/generate.config
```

### Training a model

To train a model from scratch, you must first download the full Crossdocked2020 data set:

```bash
sh download_data.sh
```

More info about this data set can be found [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020).

Then you can run the `train.py` script with the default configuration file:

```bash
python3 train.py config/train.config
```

## Citation

Please make sure to cite this work if you find it useful:

```bibtex
@article{ragoza2022chemsci,
	title={{Generating 3D molecules conditional on receptor binding sites with deep generative models}},
	author={Matthew Ragoza and Tomohide Masuda and David Ryan Koes},
	journal={Chem Sci},
	month={Feb},
	day={7},
	year={2022},
	volume={13},
	pages={2701--2713},
	doi={10.1039/D1SC05976A},
	abstract={The goal of structure-based drug discovery is to find small molecules that bind to a given target protein. Deep learning has been used to generate drug-like molecules with certain cheminformatic properties, but has not yet been applied to generating 3D molecules predicted to bind to proteins by sampling the conditional distribution of protein–ligand binding interactions. In this work, we describe for the first time a deep learning system for generating 3D molecular structures conditioned on a receptor binding site. We approach the problem using a conditional variational autoencoder trained on an atomic density grid representation of cross-docked protein–ligand structures. We apply atom fitting and bond inference procedures to construct valid molecular conformations from generated atomic densities. We evaluate the properties of the generated molecules and demonstrate that they change significantly when conditioned on mutated receptors. We also explore the latent space learned by our generative model using sampling and interpolation techniques. This work opens the door for end-to-end prediction of stable bioactive molecules from protein structures with deep learning.},
}
```
