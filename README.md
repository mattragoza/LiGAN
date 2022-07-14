# LiGAN: Generative models of molecular grids

![LiGAN methods diagram](ligan_methods.png)

LiGAN is a research project for structure-based drug discovery with deep generative models of atomic density grids. It is based on [pytorch](https://github.com/pytorch/pytorch) and [molgrid](https://github.com/gnina/libmolgrid) and makes extensive use of [rdkit](https://github.com/rdkit/rdkit) and [openbabel](https://github.com/openbabel/openbabel), as well.

To read more on the methods involved and evaluations of this work:

| Title       | Venue       | Paper       | Video        |
| ----------- | ----------- | ----------- | ------------ |
| Generating 3D molecules conditional on receptor binding sites with deep generative models | Chemical Science, February 2022 | [Link](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a)       |  |
| Generating 3D Molecular Structures Conditional on a Receptor Binding Site with Deep Generative Models | NeurIPS 2020 workshop | [Link](https://arxiv.org/abs/2010.14442) | [Link](https://youtu.be/zru1FqCd8Ks) |
| Learning a Continuous Representation of 3D Molecular Structures with Deep Generative Models | NeurIPS 2020 workshop | [Link](https://arxiv.org/abs/2010.08687)         | [Link](https://youtu.be/Pyc6xwtGaUM) |

**NOTE:** You can access the Caffe code that was used for the NeurIPS 2020 workshop by visiting the `caffe` branch.

## Installation

LiGAN is a python package, but it depends on molgrid, which is written in C++/CUDA.

To ensure compatibility with openbabel and rdkit, you should use the provided `environment.yml` to create a conda environment, then build molgrid from source within that environment.

**NOTE**: I *highly* recommend using [mamba](https://mamba.readthedocs.io/en/latest/index.html) instead of vanilla conda for managing your conda environments.
Mamba is a drop-in replacement for conda that is:

- Faster at solving environments (>10x in my experience)
- Better at resolving conflicts
- More informative when something goes wrong.

### Step 1. Create `LiGAN` conda environment

Run the following to create a conda environment containing all the dependencies needed to build molgrid and use LiGAN:

```bash
git clone git@github.com:mattragoza/LiGAN.git
mamba env create --file LiGAN/environment.yml
mamba activate LiGAN
```

### Step 2. Build libmolgrid from source

Make sure you run the following from within the `LiGAN` conda environment.

```bash
git clone git@github.com:gnina/libmolgrid.git
mkdir libmolgrid/build
cd libmolgrid/build
cmake .. \
	-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
	-DOPENBABEL3_INCLUDE_DIR=$CONDA_PREFIX/include/openbabel3 \
	-DOPENBABEL3_LIBRARIES=$CONDA_PREFIX/lib/libopenbabel.so \
	-DZLIB_LIBRARY=$CONDA_PREFIX/lib/libz.so
make -j8
make install
```
And that's it!

**NOTE:** The current version of molgrid provided through pip/conda is incompatible with conda openbabel, and you will likely get segmentation faults or indefinite hangs if you install them both through conda. There may also be version conflicts with RDKit. This is why you should use mamba and build molgrid from source.

### Step 3. Run LiGAN unit tests (optional)

```
cd ../../LiGAN
pytest tests
```

## Usage

### Configuration

The two main scripts for using this package are `train.py` and `generate.py`, which both take a config file as the only command line argument. The config files for each script use YAML format to specify settings like what data and model architecture to use.

You can change the input data used by `generate.py` and the number of type of generated molecules by editting the following config fields in `config/generate.config`:

```yaml
data:
  data_root: <path to directory containing structure files>
  data_file: <path to file that references structure files>
  ...

generate:
  n_examples: <number of receptor-ligand pairs in data file>
  n_samples: <number of generated samples per input example>
  prior: <false for posterior, true for prior>
  var_factor: <variability factor; default=1.0>
  post_factor: <interpolation factor; default=1.0>
  ...

```

The data file should be a CSV-like file where each row is a receptor-ligand pair of structure files formatted like so:

```python
{class_label:d} {affinity:f} {RMSD:f} {receptor_path} {ligand_path}
```

The full structure paths that will be loaded are `data_root/receptor_path` and `data_root/ligand_path`. You can leave the other fields (`class_label`, `affinity`, and `RMSD`) as zero if they are not known, since they are not currently used by the default model.

### Generating molecules from pretrained model

To generate molecules, you must first download the pretrained model weights:

```bash
sh download_weights.sh
```

Then just run the `generate.py` script with the default configuration file:

```bash
python3 generate.py config/generate.config
```

### Training a generative model

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
