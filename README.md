# What is liGAN?

liGAN is a research codebase for training and evaluating deep generative models for *de novo* drug design based on 3D atomic density grids. It is based on [libmolgrid](https://github.com/gnina/libmolgrid) and the [gnina](https://github.com/gnina/gnina) fork of [caffe](https://github.com/BVLC/caffe). It includes scripts for creating model architectures and job scripts from parameter files, as well as for submitting and managing experiments on Slurm or Torque-based computing clusters.

## Dependencies

- numpy
- pandas
- scikit-image
- protobuf
- torch
- rdkit
- openbabel
- molgrid
- [gnina](https://github.com/gnina/gnina) version of caffe

## Tutorial

Here is basic walkthrough on how to use the liGAN scripts to launch a training experiment. All paths in the following commands are relative to the `tutorial` directory, so run this first:

`cd <LIGAN_DIR>/tutorial`

### Specifying parameters

A few scripts in this project take a "params file" as an argument. These are simple text files where each line assigns a value to a parameter in Python-like syntax. The values must be Python literals, and the meaning of the parameters depends on which script the params file is created for.

Here is an example params file that creates some model architectures when provided to the models.py script:
```
encode_type = ['_l-l', '_vl-l']
rec_map = '/net/pulsar/home/koes/mtr22/gan/my_rec_map'
lig_map = '/net/pulsar/home/koes/mtr22/gan/my_lig_map'
data_dim = 48
resolution = 0.5
data_options = ''
n_levels = 4
conv_per_level = 3
arch_options = 'l'
n_filters = 32
width_factor = 2
n_latent = 1024
loss_types = 'e'
loss_weight_KL = 0.1
loss_weight_L2 = 1.0
```
Any parameter can be assigned a list of values instead of a single value. In that case, the params file represents every possible combination of parameter assignments.

For example, in the above file, the `encode_type` parameter is assigned two values, so two model architectures can be created from the file- a standard autoencoder and a variational autoencoder.

The `encode_type` syntax allows the following architectures to be created, and more:

```
_l-l   -> ligand autoencoder (AE)
_vl-l  -> ligand variational autoencoder (VAE)
_r-l   -> receptor-to-ligand context encoder (CE)
_rvl-l -> receptor-conditional VAE (CVAE)
```

### Creating models

The models.py script is used to create a model architecture file for each parameter assignment in the provided params file.

```
usage: models.py [-h] -o OUT_DIR [-n MODEL_NAME] [-m MODEL_TYPE] [-v VERSION]
                 [--scaffold] [--benchmark BENCHMARK] [--verbose] [--gpu]
                 params_file

Create model prototxt files from model params

positional arguments:
  params_file           file defining model params or dimensions of param
                        space

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out_dir OUT_DIR
                        common output directory for model files
  -n MODEL_NAME, --model_name MODEL_NAME
                        custom model name format
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        model type, for default model name format (e.g. data,
                        gen, or disc)
  -v VERSION, --version VERSION
                        version, for default model name format (e.g. 13,
                        default most recent)
  --scaffold            attempt to scaffold models in Caffe and estimate
                        memory usage
  --benchmark BENCHMARK
                        benchmark N forward-backward pass times and actual
                        memory usage
  --verbose             print out more info for debugging prototxt creation
  --gpu                 if benchmarking, use the GPU
```

The created files are named according to the `--model_name` format string that by passing the model params to `str.format` in python.

Be careful not to underspecify the parameters in the model name format- otherwise multiple models with the same name might be created, overwriting each other.

The following command will create the two generative models described by the params file in the previous section:

`python3 ../models.py gen_model.params -o models -n gen_{encode_type}`

For training a GAN, you will also need a data-producing model and a discriminative model:

`python3 ../models.py data_model.params -o models -n data`

`python3 ../models.py disc_model.params -o models -n disc`

### Creating solvers

To train a model with Caffe, training hyperparameters must be listed in a solver file. These can be created with solvers.py.

```
usage: solvers.py [-h] -o OUT_DIR -n SOLVER_NAME params_file

Create solver prototxt files from solver params

positional arguments:
  params_file           file defining solver params or dimensions of param
                        space

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out_dir OUT_DIR
                        common output directory for solver files
  -n SOLVER_NAME, --solver_name SOLVER_NAME
                        solver name format
```
Similar to models.py, this script creates a solver file for each parameter assignment in the params file, and again the files are named according to a name format.

Run this command to create a solver file for training with the Adam optimizer:

`python3 ../solvers.py solver.params -o solvers -n adam0`

### Creating training job scripts

For executing training jobs on a computer cluster using Slurm or Torque, you can use job_scripts.py to create a collection of job scripts ready to submit.

```
usage: job_scripts.py [-h] -t TEMPLATE [-o OUT_DIR] -n JOB_NAME params_file

Create job scripts from a template and job params

positional arguments:
  params_file           file defining job params or dimensions of param space

optional arguments:
  -h, --help            show this help message and exit
  -t TEMPLATE, --template TEMPLATE
                        job script template file
  -o OUT_DIR, --out_dir OUT_DIR
                        common directory for job working directories
  -n JOB_NAME, --job_name JOB_NAME
                        job name format
```
This fills in placeholder values in the template job script with each set of parameter assignments. Just as in the models and solvers scripts, the parameter ranges are provided as a params file.

The result is that a working directory is created for each job, named according to the `--job_name` format string. The created directories each contain a job script based on the template script where the placeholder values have been replaced with a parameter assignment from the job params file.

This command creates a job script to train each of the two generative models we've created so far:

`python3 ../job_scripts.py job.params -t csb_train.sh -n train_{gen_model_name}`

NOTE: For convenience, all of the above commands that use parameter files to setup a training experiment are contained in a single bash script, `setup.sh`.

### Submitting jobs

Once you've created the required models, solvers and job scripts, you can easily submit them to the CSB department cluster:

`python3 ../submit_job.py */csb_train.sh`

This command is also contained in `submit.sh`.

### Checking job errors

`python3 ../job_errors.py */csb_train.sh --print_errors`

This command is also contained in `errors.sh`.

### Collecting job output

`python3 ../job_errors.py */csb_train.sh --output_file tutorial.training_output`

This command is also contained in `output.sh`.
