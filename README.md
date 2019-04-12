# What is liGAN?

liGAN is a python environment for training and evaluating deep generative models for *de novo* ligand design using [gnina](https://github.com/gnina/gnina), which is based on a fork of [caffe](https://github.com/BVLC/caffe). It includes scripts for creating model, solvers, and job scripts from files specifying sets of parameters, for submitting multiple batch jobs to a queuing system and monitoring them as a single experiment, for using trained models to generate novel ligand densities and fit them with atomic structures, and for visualizing ligand densities and plotting experiment results.

## Tutorial

### Specifying parameters

A few scripts in this project take a "params file" as an argument. These are simple text files where each line assigns a value to a parameter in Python-like syntax. The values must be Python literals, and the meaning of the parameters depends on which script the params file is created for.

Here is an example params file that creates a particular model architecture when provided to the models.py script:

```
encode_type = '_vl-l'
data_dim = 24
n_levels = 3
conv_per_level = 2
arch_options = 'l'
n_filters = 32
width_factor = 2
n_latent = 1024
loss_types = 'e'
```
Any parameter can instead be assigned a list of values instead of a single value. In that case, the params file represents the set of parameter assignments formed by taking the Cartesian product of the values assigned to each parameter.

For example, in the above file, the second to last line could be changed to `n_latent = [1024, 2048]`. Then the file represents a parameter space consisting of two model architectures, each with a different latent space size (and all other parameters identical).


### Creating models

As stated above, the models.py script is used to create model architecture files. Its usage is as follows:
```
usage: models.py [-h] [-n MODEL_NAME] [-m MODEL_TYPE] [-v VERSION] [-s]
                 [-o OUT_PREFIX] [--gpu]
                 params_file

Create model prototxt files

positional arguments:
  params_file           file defining model params or dimensions of param
                        space

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model_name MODEL_NAME
                        custom model name format
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        model name format type (data, gen, or disc)
  -v VERSION, --version VERSION
                        model name format version (e.g. 13, default most
                        recent)
  -s, --scaffold        do Caffe model scaffolding
  -o OUT_PREFIX, --out_prefix OUT_PREFIX
                        common output prefix for model files
  --gpu

```
This script creates a model architecture file for each parameter assignment in the params file. The created files are named according to a name format, which can either be set explicitly with the -n argument, or a default format for a certain model type can be used with the -m argument.

Name formats are simply strings that are formatted with the parameters used to create the model. For example, the current default format for the 'gen' model type is as follows:

`
{encode_type}e13_{data_dim:d}_{resolution:g}{data_options}_{n_levels:d}_{conv_per_level:d}{arch_options}_{n_filters:d}_{width_factor:d}_{n_latent:d}_{loss_types}`

This allows models to be differentiated by their name. If a custom name format is used instead, one should be careful not to underspecify the parameters- otherwise multiple models with the same name might be created, overwriting each other.

The following command will create the model file described by the params file in the previous section:

`python models.py tutorial/gen_model.params -n example_gen -o tutorial`


### Creating solvers

todo

### Creating job scripts

todo

### Submitting jobs

todo

### Monitoring job status

todo