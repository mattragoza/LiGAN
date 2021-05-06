# What is liGAN?

liGAN is a research codebase for training and evaluating deep generative models for *de novo* drug design based on 3D atomic density grids. It is based on [libmolgrid](https://github.com/gnina/libmolgrid) and the [gnina](https://github.com/gnina/gnina) fork of [caffe](https://github.com/BVLC/caffe).

[VAE paper](https://arxiv.org/abs/2010.08687) - [2 minute talk](https://youtu.be/Pyc6xwtGaUM)

[CVAE paper](https://arxiv.org/abs/2010.14442) - [15 minute talk](https://youtu.be/zru1FqCd8Ks)

## Dependencies

- numpy
- pandas
- scikit-image
- openbabel
- rdkit
- molgrid
- torch
- protobuf
- [gnina](https://github.com/gnina/gnina) version of caffe

## Usage

The script `generate.py` is used to generate atomic density grids and molecular structures from
a trained generative model.

Its basic usage can be seen in the scripts `generate_vae.sh`:

```
LIG_FILE=$1 # e.g. data/molport/0/102906000_8.sdf

python3 generate.py \
  --data_model_file models/data_48_0.5_molport.model \
  --gen_model_file models/vae.model \
  --gen_weights_file weights/gen_e_0.1_1_disc_x_10_0.molportFULL_rand_.0.0_gen_iter_100000.caffemodel \
  --rec_file data/molport/10gs_rec.pdb \
  --lig_file $LIG_FILE \
  --out_prefix VAE \
  --n_samples 10 \
  --fit_atoms \
  --dkoes_make_mol \
  --output_sdf \
  --output_dx \
  --gpu

```

And `generate_cvae.sh`:

```
REC_FILE=$1 # e.g. data/crossdock2020/PARP1_HUMAN_775_1012_0/2rd6_A_rec.pdb
LIG_FILE=$2 # e.g. data/crossdock2020/PARP1_HUMAN_775_1012_0/2rd6_A_rec_2rd6_78p_lig_tt_min.sdf

python3 generate.py \
  --data_model_file models/data_48_0.5_crossdock.model \
  --gen_model_file models/cvae.model \
  --gen_weights_file weights/lessskip_crossdocked_increased_1.lowrmsd.0_gen_iter_1500000.caffemodel \
  --rec_file $REC_FILE \
  --lig_file $LIG_FILE \
  --out_prefix CVAE \
  --n_samples 10 \
  --fit_atoms \
  --dkoes_make_mol \
  --output_sdf \
  --output_dx \
  --gpu

```

Both scripts can be run from the root directory of the repository. 