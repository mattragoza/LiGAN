# What is liGAN?

liGAN is a python environment for training and evaluating deep generative models for *de novo* ligand design using [gnina](https://github.com/gnina/gnina), which is based on a fork of [caffe](https://github.com/BVLC/caffe). It includes scripts for creating models, solvers, and job scripts from files specifying sets of parameters, for submitting multiple batch jobs to a high-performance computing cluster and monitoring them as a single experiment, for using trained models to generate novel ligand densities and fit them with atomic structures, and for visualizing ligand densities and plotting experiment results.

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
n_latent = [1024, 2048]
loss_types = 'e'
```
Any parameter can instead be assigned a list of values instead of a single value. In that case, the params file represents every possible combination of parameter assignments.

For example, in the above file, the `n_latent` parameter is assigned two values, so two model architectures can be created from the file- each with a different latent space size and all other parameters identical.

### Creating models

As stated above, the models.py script is used to create model architecture files. Its usage is as follows:
```
usage: models.py [-h] -o OUT_DIR [-n MODEL_NAME] [-m MODEL_TYPE] [-v VERSION]
                 [-s] [--gpu]
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
  -s, --scaffold        attempt to scaffold models in Caffe
  --gpu                 if scaffolding, use the GPU
```
This script creates a model architecture file for each parameter assignment in the params file. The created files are named according to a name format, which can either be set explicitly with the -n argument, or a default format for a certain model type can be used by passing the -m argument.

Name formats are simply strings that are formatted with the parameters used to create the model. For example, the current default format for the 'gen' model type is as follows:

`{encode_type}e13_{data_dim:d}_{resolution:g}{data_options}_{n_levels:d}_{conv_per_level:d}{arch_options}_{n_filters:d}_{width_factor:d}_{n_latent:d}_{loss_types}`

This allows models to be differentiated by their name. If a custom name format is used, be careful not to underspecify the parameters- otherwise multiple models with the same name might be created, overwriting each other.

The following command will create the two generative models described by the params file in the previous section:

`python models.py tutorial/gen_model.params -o tutorial/models -n gen{n_latent}`

For training a GAN, you will also need a data-producing model and a discriminative model:

`python models.py tutorial/data_model.params -o tutorial/models -n data`

`python models.py tutorial/disc_model.params -o tutorial/models -n disc`

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

`python solvers.py tutorial/solver.params -o tutorial/solvers -n adam0`

### Creating job scripts

For executing jobs on a computer cluster using Torque or Slurm, you can use job_scripts.py to create a collection of job scripts ready to be submit.

```
usage: job_scripts.py [-h] -b JOB_TEMPLATE -o OUT_DIR -n JOB_NAME params_file

Create job scripts from a template and job params

positional arguments:
  params_file           file defining job params or dimensions of param space

optional arguments:
  -h, --help            show this help message and exit
  -b JOB_TEMPLATE, --job_template JOB_TEMPLATE
                        job script template file
  -o OUT_DIR, --out_dir OUT_DIR
                        common directory for job working directories
  -n JOB_NAME, --job_name JOB_NAME
                        job name format
```
This fills in placeholder values in a template job script with each set of parameter assignments. Just as in the models and solvers scripts, the parameter ranges are provided as a params file, and the individual jobs are named according to a name format. Some basic job template scripts are included in the job_templates sub directory, or they can be tailored to your needs.

A slight difference in this script is that the name format string is used to create a working directory for the job to run in rather than to name the job script itself. The parameterized job scripts are each created in their own working directory.

This command creates a job script to train each of the two generative models we've created so far:

`python job_scripts.py tutorial/job.params -b job_templates/slurm_train.sh -o tutorial -n {gen_model_name}`

### Submitting jobs

Once you've created a set of job scripts, you can easily submit them to a queue:

### Monitoring job status

todo