import sys, os, yaml, torch, copy
sys.path.append(os.environ['LIGAN_ROOT'])
import liGAN

GB = 1024**3


def get_cuda_memory(config, batch_size):
    config = copy.deepcopy(config)
    config['data']['batch_size'] = batch_size
    torch.cuda.reset_max_memory_allocated()

    solver_type = getattr(
        liGAN.training, config['model_type'] + 'Solver'
    )
    solver = solver_type(
        train_file=config['data'].pop('train_file'),
        test_file=config['data'].pop('test_file'),
        data_kws=config['data'],
        gen_model_kws=config['gen_model'],
        disc_model_kws=config.get('disc_model', None),
        loss_fn_kws=config['loss_fn'],
        gen_optim_kws=config['gen_optim'],
        disc_optim_kws=config.get('disc_optim', None),
        atom_fitting_kws=config['atom_fitting'],
        bond_adding_kws=config.get('bond_adding', {}),
        out_prefix=config['out_prefix'],
        caffe_init=config['caffe_init'],
        balance=config['balance'],
        device='cuda',
    )
    solver.test(n_batches=1)

    return torch.cuda.max_memory_allocated()


with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

config['out_prefix'] = 'MEMORY'

# set the initial batch_size for search
min_batch_size = 8
batch_size = min_batch_size
bszs, mems = [], []

# find largest "coarse" batch_size
# this will be min_batch_size * a power of 2
try:
    while True:
        memory = get_cuda_memory(config, batch_size)
        bszs.append(batch_size)
        mems.append(memory)
        batch_size *= 2
except RuntimeError: # CUDA out of memory
    batch_size //= 2 # undo last increase

# now know that ideal size is in [batch_size, 2*batch_size]
# so perform a binary search in this range
increment = batch_size // 2
sign = 1

# find largest "refined" batch_size
# this will be a multiple of min_batch_size
while increment >= min_batch_size:
    batch_size += sign * increment
    try:
        memory = get_cuda_memory(config, batch_size)
        bszs.append(batch_size)
        mems.append(memory)
        sign = 1 # increment on next step
    except RuntimeError: # CUDA out of memory
        sign = -1 # decrement on next step
    increment //= 2 # reduce step size

for bsz, mem in zip(bszs, mems):
    print('{} {:.2f}gb'.format(bsz, mem / GB))

