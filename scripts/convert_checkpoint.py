import sys, os, re, torch
from collections import OrderedDict as odict

layer_map = {
    "input_encoder.fc0.fc.weight": "input_encoder.fc0.1.weight",
    "input_encoder.fc0.fc.bias": "input_encoder.fc0.1.bias",
    "input_encoder.fc1.fc.weight": "input_encoder.fc1.1.weight",
    "input_encoder.fc1.fc.bias": "input_encoder.fc1.1.bias",
    "conditional_encoder.fc0.fc.weight": "conditional_encoder.fc0.1.weight",
    "conditional_encoder.fc0.fc.bias": "conditional_encoder.fc0.1.bias",
    "decoder.fc.fc.weight": "decoder.fc.0.weight",
    "decoder.fc.fc.bias": "decoder.fc.0.bias",
}

def stdize_gen_layers(state_dict):
    for old_key in list(state_dict.keys()):
        if old_key in layer_map:
            new_key = layer_map[old_key]
            state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


for in_file in sys.argv[1:]:

    prefix = os.path.splitext(in_file)[0]
    curr_iter = int(re.match(r'.*_iter_(\d+)', prefix).group(1))
    d = torch.load(in_file, map_location='cpu')

    save = lambda o,f: (print(f) or torch.save(o, f))

    save(
        stdize_gen_layers(d['gen_model_state']),
        prefix + '.gen_model_state'
    )
    save(odict([
        ('optim_state', stdize_gen_layers(d['gen_optim_state'])),
        ('iter', curr_iter),
    ]), prefix + '.gen_solver_state')

    if 'disc_model_state' in d:
        save(
            d['disc_model_state'],
            prefix + '.disc_model_state'
        )
        save(odict([
            ('optim_state', d['disc_optim_state']),
            ('iter', curr_iter),
        ]), prefix + '.disc_solver_state')

