## LiGAN configuration settings

The training and generation scripts are controlled by configuration files. The configuration settings are documented below.

### generate.config

```
# general settings
out_prefix: <str: common prefix for output files>
model_type: <str: type of generative model architecture>
random_seed: <int: seed for random number generation>
verbose: <bool: use verbose output>
device: <{cuda, cpu}: device to run model>

data: # settings for data loading and gridding
  data_root: <str: path to directory containing receptor/ligand files>
  data_file: <str: path to data file specifying receptor/ligand paths>
  batch_size: <int: number of examples per batch>
  rec_typer: <str: receptor atom typing scheme>
  lig_typer: <str: ligand atom typing scheme>
  use_rec_elems: <bool: use different element set for receptors>
  resolution: <float: grid resolution in angstroms>
  grid_size: <int: number of grid points per dimension>
  shuffle: <bool: randomly shuffle data examples>
  random_rotation: <bool: apply uniform random grid rotations>
  random_translation: <float: maximum random grid translation in angstroms>
  diff_cond_transform: <bool: apply different random transform to conditional branch>
  diff_cond_structs: <bool: use different (rec,lig) structures for conditional branch>

gen_model: # generative model architecture; must be the same as was used for training!
  n_filters: <int: number of filters in first convolution layer>
  width_factor: <int: factor by which to increase # filters in each level>
  n_levels: <int: number of convolution blocks with pooling layers between>
  conv_per_level: <int: number of convolution layers per block/level>
  kernel_size: <int: convolution kernel size>
  relu_leak: <float: slope for leaky ReLU activation function>
  batch_norm: <int: flag for including batch normalization layers>
  spectral_norm: <int: flag for including spectral normalization layers>
  pool_type: <str: type of pooling layers (a=average, m=max, c=conv)>
  unpool_type: <str: type of upsampling layers (n=nearest, c=conv)>
  pool_factor: <int: kernel size and stride for pooling layers>
  n_latent: <int: width of latent space>
  init_conv_pool: <int: flag for including a separate initial conv/pool layer pair>
  skip_connect: <bool: flag for conditional skip connections from encoder to decoder>
  block_type: <str: type of convolution block (c=standard, r=residal, d=dense)>
  growth_rate: <int: growth rate for dense convolution blocks (no effect otherwise)>
  bottleneck_factor: <int: bottleneck factor for dense blocks (no effect otherwise)>
  state: <str: path to generative model state file containing trained weights>

atom_fitting: # atom fitting settings
  beam_size: <int: number of top-ranked structures to maintain during search>
  multi_atom: <bool: allow placing multiple detected atoms simultaneously>
  n_atoms_detect: <int: number of top-ranked atoms to detect at each step>
  apply_conv: <bool: apply convolution with density kernel before detecting atoms>
  threshold: <float: threshold for detecting atoms in residual density>
  peak_value: <float: upper limit for detecting atoms in residual density>
  min_dist: <float: minimum distance between detected atoms>
  apply_prop_conv: <bool: apply convolution to atom property channels>
  interm_gd_iters: <int: number of gradient descent steps after each atom placement>
  final_gd_iters: <int: number of gradient descent steps on final structure>

generate: # molecule generation settings
  n_examples: <int: number of examples from input data to generate>
  n_samples: <int: number of samples to generate per input example>
  prior: <bool: sample from prior distribution instead of posterior>
  var_factor: <float: variability factor; controls sample diversity>
  post_factor: <float: posterior factor; controls similarity to input>
  stage2: <bool: use stage2 VAE; experimental feature>
  truncate: <bool: truncate tails of latent sampling distribution>
  interpolate: <bool: latent interpolation; experimental feature>
  spherical: <bool: use spherical latent interpolation>
  fit_atoms: <bool: fit atoms to generated densities>
  add_bonds: <bool: add bonds to generate molecules from fit atoms>
  uff_minimize: <bool: minimize internal energy of generated molecules>
  gnina_minimize: <bool: minimize generated molecules in receptor pocket>
  minimize_real: <bool: minimize input molecules for comparison>

output:
  batch_metrics: <bool: compute batch-level metrics>
  output_grids: <bool: write generated densities to .dx files (these are large)>
  output_structs: <bool: write fit atom structures to .sdf files>
  output_mols: <bool: write generated molecules to .sdf files>
  output_latents: <bool: write sampled latent vectors to .latent files>
  output_visited: <bool: include all visited atom fitting structures>
  output_conv: <bool: write atom density kernel to .dx files>
```

### train.config

TODO
